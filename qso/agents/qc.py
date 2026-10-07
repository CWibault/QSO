import copy
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from qso.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from qso.utils.networks import GCActorVectorField, GCValue

class QCAgent(flax.struct.PyTreeNode):
    """
    RL with Action Chunking.
    Implements both IQL for value learning (conservative) and standard (non-conservative) Q-function value learning. 
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

    def iql_value_loss(self, batch, grad_params):
        """
        Compute the conservative (IQL) value loss.
        """
        batch_action_chunks = jnp.reshape(batch["action_chunks"], (batch["action_chunks"].shape[0], -1))
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], 
            batch["high_goals"], 
            batch_action_chunks
        )
        q = jnp.minimum(q1, q2)

        v = self.network.select("value")(
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

    def iql_critic_loss(self, batch, grad_params):
        """
        Compute the conservative (IQL) critic loss.
        """
        next_v = self.network.select("value")(
            batch["high_subgoals"], 
            batch["high_goals"]
        )
        q = batch["high_rewards_synthetic"] + batch["high_discounts_synthetic"] * batch["high_masks"] * next_v

        batch_action_chunks = jnp.reshape(batch["action_chunks"], (batch["action_chunks"].shape[0], -1))
        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["high_goals"],
            batch_action_chunks,
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
    
    def critic_loss(
        self, 
        batch, 
        grad_params, 
        rng
        ):
        """
        Compute the non-conservative critic loss.
        """
        batch_action_chunks = jnp.reshape(batch["action_chunks"], (batch["action_chunks"].shape[0], -1))
        
        rng, sample_rng = jax.random.split(rng)

        next_action_chunks = self.sample_actions(
            batch["high_subgoals"],
            batch["high_goals"],
            seed=sample_rng,
            num_samples=self.config["critic_num_samples"],
            )
        next_qs = self.network.select("target_critic")(
            batch["high_subgoals"], 
            batch["high_goals"], 
            next_action_chunks
        )
        next_q = next_qs.mean(axis=0)
        target_q = batch["high_rewards_synthetic"] + batch["high_discounts_synthetic"] * batch["high_masks"] * next_q

        q = self.network.select("critic")(
            batch["observations"],
            batch["high_goals"],
            actions=batch_action_chunks, 
            params=grad_params
            )
        
        critic_loss = ((q - target_q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """
        Compute the BC flow loss.
        """
        batch_action_chunks = jnp.reshape(batch["action_chunks"], (batch["action_chunks"].shape[0], -1))
        batch_size, action_dim = batch_action_chunks.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # --- bc flow loss. --- 
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_action_chunks
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select("actor_bc_flow")(
            batch["observations"],
            batch["high_goals"],
            x_t,
            t,
            params=grad_params)

        # --- only bc on the non-synthetic chunk indices. ---
        sq_err = jnp.reshape(
            (pred - vel) ** 2,
            (batch_size, self.config["subgoal_steps"], self.config["action_dim"]),
        )
        mask = (~batch["synthetics"]) if self.config["actor_no_synthetics"] else jnp.ones_like(batch["synthetics"])
        bc_flow_loss = jnp.sum(sq_err * mask[..., None]) / (jnp.maximum(jnp.sum(mask), 1) * self.config["action_dim"])

        return bc_flow_loss, {
            "bc_flow_loss": bc_flow_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """
        Compute the total loss.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self.config["conservative_value_learning"] == True:
            value_loss, value_info = self.iql_value_loss(batch, grad_params)
            for k, v in value_info.items():
                info[f"value/{k}"] = v

            critic_loss, critic_info = self.iql_critic_loss(batch, grad_params)
            for k, v in critic_info.items():
                info[f"critic/{k}"] = v
        
            critic_loss = value_loss + critic_loss
        
        else: 
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
            for k, v in critic_info.items():
                info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
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
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @functools.partial(jax.jit, static_argnames=("num_samples",))
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        num_samples=None,
        temperature=1.0,
    ):

        num_samples = self.config["actor_num_samples"] if num_samples is None else num_samples
        action_dim = self.config["action_dim"] * self.config["subgoal_steps"]
        noises = jax.random.normal(seed,
                (*observations.shape[: -len(self.config["observation_dim"])],num_samples, action_dim))

        # --- repeat conditioning inputs over "best-of-n" samples. ---
        observations = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
        goals = jnp.repeat(goals[..., None, :], num_samples, axis=-2)
        actions = self.compute_flow_actions(
            observations,
            goals,
            noises)
        q = self.network.select("critic")(observations, goals, actions).min(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, num_samples, action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        goals,
        noises,
    ):
        """
        Compute actions from the BC flow model using the Euler method.
        """
        actions = noises
        for i in range(self.config["actor_flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["actor_flow_steps"])
            vels = self.network.select("actor_bc_flow")(observations, goals, actions, t)
            actions = actions + vels / self.config["actor_flow_steps"]
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(
        cls,
        seed,
        ex_batch,
        config,
    ):
        """
        Create a new agent.

        Args:
            seed: Random seed.
            ex_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch["observations"]
        ex_actions = ex_batch["actions"]
        ex_goals = ex_batch["high_goals"]
        ex_times = ex_actions[..., :1]
        ex_full_actions = jnp.concatenate([ex_actions] * config["subgoal_steps"], axis=-1)
        
        observation_dim = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        full_action_dim = ex_full_actions.shape[-1]

        # --- define networks ---
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=1,
        )
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=2,
        )
        actor_bc_flow_def = GCActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_full_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, ex_full_actions, ex_times)),
        )
        if config["conservative_value_learning"]:
            network_info.update(
                value=(value_def, (ex_observations, ex_goals)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params[f"modules_target_critic"] = params[f"modules_critic"]

        config["observation_dim"] = observation_dim
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            # --- environment hyperparameters. --- 
            observation_dim=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            # --- agent hyperparameters. ---
            agent_name="qc",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size. 
            # --- value hyperparameters. --- 
            conservative_value_learning=False, # Whether to use IQL. 
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_tau=0.005,  # Target network update rate.
            value_expectile=0.7,  # IQL expectile.
            # --- actor hyperparameters. --- 
            actor_no_synthetics=True, # Whether to only do the bc on the non-synthetic actions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_flow_steps=6,  # Number of flow steps.
            critic_num_samples=32,  # "Best-of-n" for the TD target.
            actor_num_samples=32,  # "Best-of-n" at action-selection time.
            # --- dataset hyperparameters. ---
            dataset_class="HRGCDataset",  # Dataset class name.
            discount=0.99,  # Discount factor.
            high_env_reward_relabel=False, # Whether to use lenient reward relabelling. 
            high_p_curgoal=0.1,  # Probability of using the current state as the value goal.
            high_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            high_p_randomgoal=0.4,  # Probability of using a random state as the value goal.
            subgoal_steps=5,  # Actor subgoal steps.
            hierarchical_agent=False, # Whether goals need to be sampled for a low-level value function. 
            low_env_reward_relabel=False, # Whether to use lenient reward relabelling. 
            low_p_curgoal=0.1,  # Probability of using the current state as the value goal.
            low_p_trajgoal=0.85,  # Probability of using a future state in the same trajectory as the value goal.
            low_p_randomgoal=0.05,  # Probability of using a random state as the value goal.
        )
    )
    return config