import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from qso.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from qso.utils.networks import (
    MLP,
    GCActor,
    GCValue,
    LengthNormalize,
    length_normalize,
)

class HIQLAgent(flax.struct.PyTreeNode):
    """
    Hierarchical implicit Q-learning agent.
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """
        Compute the expectile loss.
        """
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """
        Compute the conservative (and optimistic) value loss. Goal representations are learned with the value function.
        """
        high_goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["goals"]], axis=-1),
            params = grad_params
            )
        target_high_goal_reps = self.network.select("target_goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["goals"]], axis=-1))
        target_high_next_goal_reps = self.network.select("target_goal_rep")(
            jnp.concatenate(
                [batch["next_observations"], batch["goals"]], axis=-1))

        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], 
            target_high_next_goal_reps
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t
        (v1_t, v2_t) = self.network.select("target_value")(
            batch["observations"], 
            target_high_goal_reps
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        (v1, v2) = self.network.select("value")(
            batch["observations"], 
            high_goal_reps, 
            params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["value_expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["value_expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def low_actor_loss(self, batch, grad_params, rng=None):
        """
        Compute the low-level actor loss.
        """
        # --- goal representations. ---
        goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["subgoals"]], axis=-1))
        next_goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["next_observations"], batch["subgoals"]], axis=-1))

        # --- calculate advantage. ---
        v1, v2 = self.network.select("value")(
            batch["observations"], 
            jax.lax.stop_gradient(goal_reps)
        )
        nv1, nv2 = self.network.select("value")(
            batch["next_observations"], 
            jax.lax.stop_gradient(next_goal_reps)
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        exp_a = jnp.exp(adv * self.config["low_awr_alpha"])
        exp_a = jnp.minimum(exp_a, self.config["low_awr_clip"])

        # --- actor loss. --- 
        dist = self.network.select("low_actor")(
            batch["observations"], 
            goal_reps, 
            params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }
        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params, rng=None):
        """
        Compute the high-level actor loss.
        """

        # --- representations. --- 
        goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], 
                batch["subgoals"]], axis=-1
                )
            )
        high_goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], 
                batch["goals"]], axis=-1
            )
        )
        high_next_goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["subgoals"], 
                batch["goals"]], axis=-1
            )
        )

        # --- calculate advantage. --- 
        v1, v2 = self.network.select("value")(
            batch["observations"], 
            high_goal_reps
        )
        nv1, nv2 = self.network.select("value")(
            batch["subgoals"], 
            high_next_goal_reps
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        # --- actor loss. --- 
        dist = self.network.select("high_actor")(
            batch["observations"], 
            batch["goals"], 
            params=grad_params
        )
        log_prob = dist.log_prob(goal_reps)
        exp_a = jnp.exp(adv * self.config["high_awr_alpha"])
        exp_a = jnp.minimum(exp_a, self.config["high_awr_clip"])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - goal_reps) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """
        Compute the total loss.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 3)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(
            batch, grad_params, low_actor_rng
        )
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(
            batch, grad_params, high_actor_rng
        )
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        loss = (
            value_loss
            + low_actor_loss
            + high_actor_loss
        )
        return loss, info

    def target_update(self, network, module_name):
        """
        Update the target network.
        """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["value_tau"] + tp * (1 - self.config["value_tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """
        Update the agent and return a new agent with information dictionary.
        """
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "value")
        agent.target_update(new_network, "goal_rep")

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """
        Sample actions from the actor.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select("high_actor")(
            observations, 
            goals, 
            temperature=temperature
        )
        goal_reps = length_normalize(high_dist.sample(seed=high_seed))

        low_dist = self.network.select("low_actor")(
            observations, 
            goal_reps, 
            temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["goals"]
        ex_goal_reps = jnp.zeros(shape=(ex_observations.shape[0], config["rep_dim"]))

        observation_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        goal_rep_def = nn.Sequential([
            MLP(
                hidden_dims=(*config["rep_hidden_dims"], config["rep_dim"]),
                activate_final=False
                ),
            LengthNormalize()]
            )
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=2,
        )
        target_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=2,
        )
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["rep_dim"],
        )
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            target_goal_rep=(copy.deepcopy(goal_rep_def),  (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goal_reps)),
            target_value=(copy.deepcopy(target_value_def), (ex_observations, ex_goal_reps)),
            low_actor=(low_actor_def, (ex_observations, ex_goal_reps)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_goal_rep"] = params["modules_goal_rep"]

        config["observation_dim"] = observation_dim
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # --- environment hyperparameters. --- 
            observation_dim=ml_collections.config_dict.placeholder(int),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimensions (will be set automatically).
            # --- agent hyperparameters. ---
            agent_name="hiql",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            # --- representation hyperparameters. ---
            rep_dim=128,  # Goal representation dimension.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            # --- value function hyperparameters. ---
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_tau=0.005,  # Target network update rate.
            value_expectile=0.7,  # IQL expectile.
            # --- actor hyperparameters. ---
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            low_awr_alpha=3.0,  # Low-level AWR temperature.
            low_awr_clip=100.0,  # Low-level AWR clip.
            high_awr_alpha=3.0,  # High-level AWR temperature.
            high_awr_clip=100.0,  # High-level AWR clip.
            # --- dataset hyperparameters. ---
            dataset_class="HGCDataset",  # Dataset class name.
            discount=0.995,  # Discount factor.
            env_reward_relabel=True, # Whether to use lenient reward relabelling. 
            p_curgoal=0.1,  # Probability of using the current state as the value goal.
            p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            p_randomgoal=0.4,  # Probability of using a random state as the value goal.
            subgoal_steps=25,  # Actor subgoal steps.
            )
    )
    return config
