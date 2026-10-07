import copy
import functools
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
    GCValue,
    GCActorVectorField,
    LengthNormalize,
    length_normalize,
)

class QSOAgent(flax.struct.PyTreeNode):
    """
    Abstract RL agent with a flow-matching high-level AND a flow-matching low-level policy over a learned option space (QSODF).
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

    def high_value_loss(
        self, 
        batch, 
        grad_params, 
        target_options
        ):
        """
        Compute the IQL value loss.
        """
        q1, q2 = self.network.select("target_high_critic")(
            batch["observations"], 
            batch["high_goals"], 
            target_options
        )
        q = jnp.minimum(q1, q2)

        v = self.network.select("high_value")(
            batch["observations"], 
            batch["high_goals"], 
            params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["value_expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def high_critic_loss(
        self, 
        batch, 
        grad_params, 
        options):
        """
        Compute the IQL critic loss.
        """
        next_v = self.network.select("high_value")(
            batch["option_end"], 
            batch["high_goals"]
        )
        q = batch["high_rewards_synthetic"] + batch["high_discounts_synthetic"] * batch["high_masks"] * next_v

        q1, q2 = self.network.select("high_critic")(
            batch["observations"],
            batch["high_goals"],
            options,
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        critic_info = {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "g_range": q.max() - q.min()
        }

        return critic_loss, critic_info

    def low_value_loss(
        self, 
        batch, 
        grad_params, 
        low_options,
        target_low_options
        ):
        """
        Compute the IQL value loss.
        """
        q1, q2 = self.network.select("target_low_critic")(
            batch["observations"],
            jax.lax.stop_gradient(target_low_options),
            batch["actions"]
        )
        q = jnp.minimum(q1, q2)

        v = self.network.select("low_value")(
            batch["observations"],
            jax.lax.stop_gradient(low_options),
            params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["value_expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def low_critic_loss(
        self, 
        batch, 
        grad_params, 
        low_options
        ):
        """
        Compute the IQL critic loss.
        """
        if self.config["rep_encode_ablation"]:
            next_low_options = self.network.select("option_rep")(
                batch["low_goals"],
            )
        else: 
            next_low_options = self.network.select("option_rep")(
                jnp.concatenate([batch["next_observations"], batch["low_goals"]], axis=-1),
            )
            
        next_v = self.network.select("low_value")(
            batch["next_observations"],
            next_low_options
        )
        q = batch["low_rewards"] + self.config["low_discount"] * batch["low_masks"] * next_v

        q1, q2 = self.network.select("low_critic")(
            batch["observations"],
            low_options,
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
    
    def low_actor_loss(
        self, 
        batch, 
        grad_params,
        options,
        rng
        ):
        """
        Compute the low-level actor loss.
        """
        # --- regress towards action. --- 
        batch_size, action_dim = batch["actions"].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        # --- bc flow loss. ---
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select("low_actor_bc_flow")(
            batch["observations"], 
            options, 
            x_t, 
            t, 
            params=grad_params)
        bc_flow_loss = (pred - vel) ** 2

        # --- advantage weighting. ---
        q1, q2 = self.network.select("low_critic")(
            batch["observations"],
            options,
            batch["actions"],
        )
        v = self.network.select("low_value")(
            batch["observations"],
            options,
        )
        q = jnp.minimum(q1, q2)
        adv = q - v
        exp_a = jnp.exp(adv * self.config["awr_alpha"])
        exp_a = jnp.minimum(exp_a, self.config["awr_clip"])

        # --- effective sample size of the AWR weights. ---
        eff_batch_frac = (exp_a.sum() ** 2 / (exp_a ** 2).sum()) / batch_size

        # --- final low actor loss. ---
        if self.config["low_actor_loss"] == "awr":
            actor_loss = (exp_a[:, None] * bc_flow_loss).mean()
        else: 
            actor_loss = bc_flow_loss.mean()
            
        actor_info = {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss.mean(),
            "eff_batch_frac": eff_batch_frac,
        }

        return actor_loss, actor_info

    def high_actor_loss(
        self,
        batch,
        grad_params,
        options,
        rng
        ):
        """
        Compute the high-level actor loss.
        """
        # --- regress towards option_rep. --- 
        batch_size, option_dim = options.shape
        x_rng, t_rng = jax.random.split(rng, 2)

        # --- bc flow loss. ---
        x_0 = jax.random.normal(x_rng, (batch_size, option_dim))
        x_1 = options
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select("high_actor_bc_flow")(
            batch["observations"],
            batch["high_goals"],
            x_t,
            t,
            params=grad_params)
        bc_flow_loss = (pred - vel) ** 2

        # --- advantage weighting. ---
        q1, q2 = self.network.select("high_critic")(
            batch["observations"],
            batch["high_goals"],
            options,
        )
        v = self.network.select("high_value")(
            batch["observations"],
            batch["high_goals"],
        )
        q = jnp.minimum(q1, q2)
        adv = q - v
        exp_a = jnp.exp(adv * self.config["awr_alpha"])
        exp_a = jnp.minimum(exp_a, self.config["awr_clip"])

        # --- effective sample size of the AWR weights. ---
        eff_batch_frac = (exp_a.sum() ** 2 / (exp_a ** 2).sum()) / batch_size

        # --- final high actor loss. ---
        if self.config["high_actor_loss"] == "awr":
            actor_loss = (exp_a[:, None] * bc_flow_loss).mean()
        else: 
            actor_loss = bc_flow_loss.mean()
        actor_info = {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss.mean(),
            "eff_batch_frac": eff_batch_frac,
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 3)

        # --- option representation ablations. ---
        if self.config["rep_encode_ablation"]:
            options = self.network.select("option_rep")(
                batch["option_end"],
                params=grad_params,
            )
            target_options = self.network.select("target_option_rep")(
                batch["option_end"]
            )
            low_options = self.network.select("option_rep")(
                batch["low_goals"],
                params=grad_params,
            )
            target_low_options = self.network.select("target_option_rep")(
                batch["low_goals"],
            )
        else:
            options = self.network.select("option_rep")(
                jnp.concatenate([batch["option_start"], batch["option_end"]], axis=-1),
                params=grad_params,
            )
            target_options = self.network.select("target_option_rep")(
                jnp.concatenate([batch["option_start"], batch["option_end"]], axis=-1),
            )
            low_options = self.network.select("option_rep")(
                jnp.concatenate([batch["observations"], batch["low_goals"]], axis=-1),
                params=grad_params,
            )
            target_low_options = self.network.select("target_option_rep")(
                jnp.concatenate([batch["observations"], batch["low_goals"]], axis=-1),
            )

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params, target_options)
        for k, v in high_value_info.items():
            info[f"high_value/{k}"] = v

        if self.config["rep_high_critic_stop_grad"]:
            options = jax.lax.stop_gradient(options)
        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params, options)
        for k, v in high_critic_info.items():
            info[f"high_critic/{k}"] = v

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params, low_options, target_low_options)
        for k, v in low_value_info.items():
            info[f"low_value/{k}"] = v

        if self.config["rep_low_critic_stop_grad"]: 
            low_options = jax.lax.stop_gradient(low_options)
        low_critic_loss, low_critic_info = self.low_critic_loss(batch, grad_params, low_options)
        for k, v in low_critic_info.items():
            info[f"low_critic/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, jax.lax.stop_gradient(options), high_actor_rng)
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, jax.lax.stop_gradient(options), low_actor_rng)
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v
        
        loss = (
            + high_value_loss
            + high_critic_loss
            + low_value_loss
            + low_critic_loss
            + high_actor_loss
            + low_actor_loss
        )

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        params = self.network.params[f"modules_{module_name}"]
        target_params = self.network.params[f"modules_target_{module_name}"]

        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["value_tau"] + tp * (1 - self.config["value_tau"]),
            params,
            target_params,
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "option_rep")
        agent.target_update(new_network, "high_critic")
        agent.target_update(new_network, "low_critic")

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def compute_flow_options(
        self,
        observations,
        goals,
        noises,
    ):
        """
        Compute options from the BC flow model using the Euler method.
        """
        options = noises
        # --- euler method. ---
        for i in range(self.config["actor_flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["actor_flow_steps"])
            vels = self.network.select("high_actor_bc_flow")(observations, goals, options, t)
            options = options + vels / self.config["actor_flow_steps"]
        # --- match the length normalisation the option representation applies. ---
        return length_normalize(options)
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        options,
        noises
        ):
        """
        Compute actions from the BC flow model using the Euler method.
        """
        actions = noises
        for i in range(self.config["actor_flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["actor_flow_steps"])
            vels = self.network.select("low_actor_bc_flow")(observations, options, actions, t)
            actions = actions + vels / self.config["actor_flow_steps"]
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        options=None,
        temperature=1.0,
    ):
        """Sample actions from the actor, drawing a new option only when one is not handed in."""
        high_seed, low_seed = jax.random.split(seed)

        # --- high-level rejection sampling. --- 
        n_noises = jax.random.normal(high_seed, (*observations.shape[:-1], self.config["high_actor_num_samples"], self.config["rep_dim"]))
        n_observations = jnp.repeat(observations[..., None, :], self.config["high_actor_num_samples"], axis=-2)
        n_goals = jnp.repeat(goals[..., None, :], self.config["high_actor_num_samples"], axis=-2)
        n_options = self.compute_flow_options(n_observations, n_goals, n_noises)
        n_options_q = self.network.select("high_critic")(n_observations, n_goals, n_options).min(axis=0)
        high_indices = jnp.argmax(n_options_q, axis=-1)[..., None, None]
        options = jnp.take_along_axis(n_options, high_indices, axis=-2).squeeze(-2)

        # --- low-level rejection sampling. ---
        n_noises = jax.random.normal(low_seed, (*observations.shape[:-1], self.config["low_actor_num_samples"], self.config["action_dim"]))
        n_observations = jnp.repeat(observations[..., None, :], self.config["low_actor_num_samples"], axis=-2)
        n_options = jnp.repeat(options[..., None, :], self.config["low_actor_num_samples"], axis=-2)
        n_actions = self.compute_flow_actions(n_observations, n_options, n_noises)
        n_actions_q = self.network.select("low_critic")(n_observations, n_options, n_actions).min(axis=0)
        low_indices = jnp.argmax(n_actions_q, axis=-1)[..., None, None]
        actions = jnp.take_along_axis(n_actions, low_indices, axis=-2).squeeze(-2)
        return jnp.clip(actions, -1, 1)

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
        ex_goals = example_batch["high_goals"]
        ex_times = ex_actions[..., :1]
        ex_options = jnp.zeros(shape=(1, config["rep_dim"]))

        if config["rep_encode_ablation"]:
            ex_rep_inputs = ex_goals
        else: 
            ex_rep_inputs = jnp.concatenate([ex_observations, ex_goals], axis=-1)

        observation_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        option_rep_def = nn.Sequential([
            MLP(
                hidden_dims=(*config["rep_hidden_dims"], config["rep_dim"]),
                activate_final=False,
            ),
            LengthNormalize()]
        )
        high_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=1,
        )
        high_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=2,
        )
        high_actor_bc_flow_def = GCActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["rep_dim"],
        )
        low_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=1,
        )
        low_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=2,
        )
        low_actor_bc_flow_def = GCActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
        )

        network_info = dict(
            option_rep=(option_rep_def, (ex_rep_inputs,)),
            target_option_rep=(copy.deepcopy(option_rep_def), (ex_rep_inputs,)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_options)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goals, ex_options)),
            high_actor_bc_flow=(high_actor_bc_flow_def, (ex_observations, ex_goals, ex_options, ex_times)),
            low_value=(low_value_def, (ex_observations, ex_options)),
            low_critic=(low_critic_def, (ex_observations, ex_options, ex_actions)),
            target_low_critic=(copy.deepcopy(low_critic_def), (ex_observations, ex_options, ex_actions)),
            low_actor_bc_flow=(low_actor_bc_flow_def, (ex_observations, ex_options, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_option_rep"] = params["modules_option_rep"]
        params["modules_target_high_critic"] = params["modules_high_critic"]
        params["modules_target_low_critic"] = params["modules_low_critic"]

        config["observation_dim"] = observation_dim
        config["action_dim"] = action_dim
        config["low_discount"] = 1 - 1 / config["subgoal_steps"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # --- environment hyperparameters. --- 
            observation_dim=ml_collections.config_dict.placeholder(int),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimensions (will be set automatically).
            # --- agent hyperparameters. ---
            agent_name="qso",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            # --- representation hyperparameters. ---
            rep_encode_ablation=False, # Option-rep input ablation (subgoal \phi_\omega(s_{t+k}) rather than relative \phi_\omega(s_t, s_{t+k}) (default)).
            rep_dim=10,  # Option representation dimension.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            rep_high_critic_stop_grad=False, # High Critic Gradient-Flow Ablation.
            rep_low_critic_stop_grad=False, # Low Critic Gradient-Flow Ablation.
            # --- value function hyperparameters. ---
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_tau=0.005,  # Target network update rate.
            value_expectile=0.7,  # IQL expectile.
            # --- actor hyperparameters. ---
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_flow_steps=6,  # Number of flow steps.
            awr_alpha=3.0,  # AWR temperature.
            awr_clip=100.0,  # AWR clip.
            low_actor_loss="bc", # Low actor loss. 
            low_actor_num_samples=1,  # "Best-of-n": options ranked under the critic.
            high_actor_loss="awr", # High actor loss. 
            high_actor_num_samples=32,  # "Best-of-n": options ranked under the critic.
            # --- dataset hyperparameters. ---
            dataset_class="HRGCDataset",  # Dataset class name.
            discount=0.99,  # Discount factor.
            high_env_reward_relabel=True, # Whether to use lenient reward relabelling. 
            high_p_curgoal=0.1,  # Probability of using the current state as the value goal.
            high_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            high_p_randomgoal=0.4,  # Probability of using a random state as the value goal.
            subgoal_steps=25,  # Actor subgoal steps.
            hierarchical_agent=True, # Whether goals need to be sampled for a low-level value function.
            low_env_reward_relabel=False, # Whether to use lenient reward relabelling. 
            low_p_curgoal=0.1,  # Probability of using the current state as the value goal.
            low_p_trajgoal=0.85,  # Probability of using a future state in the same trajectory as the value goal.
            low_p_randomgoal=0.05,  # Probability of using a random state as the value goal.
            low_discount=ml_collections.config_dict.placeholder(float) # set based on subgoal steps.
            )
    )
    return config