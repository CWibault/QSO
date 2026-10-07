import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from qso.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from qso.utils.networks import GCActorVectorField, GCValue

class ASOAgent(flax.struct.PyTreeNode):
    """
    Agent that learns Absolute Subgoal Options (ASO). 
    DSHARSA is ASO with high_actor_num_samples = 32, low_actor_num_samples = 32, and high_actor_loss = "bc" and low_actor_loss = "bc". 
    SHARSA is ASO with high_actor_num_samples = 32, low_actor_num_samples = 1, and high_actor_loss = "bc" and low_actor_loss = "bc". 
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

    def high_value_loss(self, batch, grad_params):
        """
        Compute the IQL value loss.
        """
        q1, q2 = self.network.select("target_high_critic")(
            batch["observations"], 
            batch["high_goals"], 
            batch["high_subgoals"]
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
        grad_params
        ):
        """
        Compute the IQL critic loss.
        """
        next_v = self.network.select("high_value")(
            batch["high_subgoals"], 
            batch["high_goals"]
        )
        q = batch["high_rewards_synthetic"] + batch["high_discounts_synthetic"] * batch["high_masks"] * next_v

        q1, q2 = self.network.select("high_critic")(
            batch["observations"],
            batch["high_goals"],
            batch["high_subgoals"],
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def low_value_loss(
        self, 
        batch, 
        grad_params
        ):
        """
        Compute the IQL value loss.
        """
        q1, q2 = self.network.select("target_low_critic")(
            batch["observations"],
            batch["low_goals"],
            batch["actions"]
        )
        q = jnp.minimum(q1, q2)

        v = self.network.select("low_value")(
            batch["observations"],
            batch["low_goals"],
            params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["value_expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def low_critic_loss(self, batch, grad_params):
        """
        Compute the IQL critic loss.
        """
        next_v = self.network.select("low_value")(
            batch["next_observations"],
            batch["low_goals"]
        )
        q = batch["low_rewards"] + self.config["low_discount"] * batch["low_masks"] * next_v

        q1, q2 = self.network.select("low_critic")(
            batch["observations"],
            batch["low_goals"],
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
        rng
        ):
        """
        Compute the low-level flow BC actor loss.
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
            batch["high_subgoals"], 
            x_t, 
            t, 
            params=grad_params)
        bc_flow_loss = (pred - vel) ** 2

        # --- final low actor loss. ---
        if self.config["low_actor_loss"] == "awr":

            q1, q2 = self.network.select("low_critic")(
                batch["observations"],
                batch["high_subgoals"],
                batch["actions"],
            )
            v = self.network.select("low_value")(
                batch["observations"],
                batch["high_subgoals"],
            )
            q = jnp.minimum(q1, q2)
            adv = q - v
            exp_a = jnp.exp(adv * self.config["awr_alpha"])
            exp_a = jnp.minimum(exp_a, self.config["awr_clip"])
            eff_batch_frac = (exp_a.sum() ** 2 / (exp_a ** 2).sum()) / batch_size # effective sample size of the AWR weights.
            actor_loss = (exp_a[:, None] * bc_flow_loss).mean()

        else: 

            eff_batch_frac = 1.0
            actor_loss = bc_flow_loss.mean()

        actor_info = {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss.mean(),
            "eff_batch_frac": eff_batch_frac
        }

        return actor_loss, actor_info

    def high_actor_loss(
        self,
        batch,
        grad_params,
        rng
        ):
        """
        Compute the high-level flow BC actor loss.
        """
        # --- regress towards subgoal. --- 
        batch_size, option_dim = batch["high_subgoals"].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        # --- bc flow loss. ---
        x_0 = jax.random.normal(x_rng, (batch_size, option_dim))
        x_1 = batch["high_subgoals"]
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

        # --- final high actor loss. ---
        if self.config["high_actor_loss"] == "awr":

            q1, q2 = self.network.select("high_critic")(
                batch["observations"],
                batch["high_goals"],
                batch["high_subgoals"],
            )
            v = self.network.select("high_value")(
                batch["observations"],
                batch["high_goals"],
            )
            q = jnp.minimum(q1, q2)
            adv = q - v
            exp_a = jnp.exp(adv * self.config["awr_alpha"])
            exp_a = jnp.minimum(exp_a, self.config["awr_clip"])
            eff_batch_frac = (exp_a.sum() ** 2 / (exp_a ** 2).sum()) / batch_size # effective sample size of the AWR weights.
            actor_loss = (exp_a[:, None] * bc_flow_loss).mean()

        else:

            eff_batch_frac = 1.0 
            actor_loss = bc_flow_loss.mean()

        actor_info = {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss.mean(),
            "eff_batch_frac": eff_batch_frac
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """
        Compute the total loss.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 3)

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params)
        for k, v in high_critic_info.items():
            info[f'high_critic/{k}'] = v

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f'low_value/{k}'] = v

        low_critic_loss, low_critic_info = self.low_critic_loss(batch, grad_params)
        for k, v in low_critic_info.items():
            info[f'low_critic/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = (
            high_value_loss
            + high_critic_loss
            + low_value_loss
            + low_critic_loss
            + high_actor_loss
            + low_actor_loss
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
        noises
        ):
        """
        Compute options from the BC flow model using the Euler method.
        """
        options = noises
        for i in range(self.config["actor_flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["actor_flow_steps"])
            vels = self.network.select("high_actor_bc_flow")(observations, goals, options, t)
            options = options + vels / self.config["actor_flow_steps"]
        return options
    
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
        temperature=1.0,
    ):
        """
        Sample actions from the actor.
        """
        high_seed, low_seed = jax.random.split(seed)

        # --- high-level rejection sampling. --- 
        n_noises = jax.random.normal(high_seed,(*observations.shape[:-1], self.config["high_actor_num_samples"], self.config["observation_dim"]))
        n_observations = jnp.repeat(observations[..., None, :], self.config["high_actor_num_samples"], axis=-2)
        n_goals = jnp.repeat(goals[..., None, :], self.config["high_actor_num_samples"], axis=-2)
        n_options = self.compute_flow_options(n_observations, n_goals, n_noises)
        n_options_q = self.network.select("high_critic")(n_observations, n_goals, n_options).min(axis=0)
        high_indices = jnp.argmax(n_options_q, axis=-1)[..., None, None]
        subgoals = jnp.take_along_axis(n_options, high_indices, axis=-2).squeeze(-2)

        # --- low-level rejection sampling. ---
        n_noises = jax.random.normal(low_seed, (*observations.shape[:-1], self.config["low_actor_num_samples"], self.config["action_dim"]))
        n_observations = jnp.repeat(observations[..., None, :], self.config["low_actor_num_samples"], axis=-2)
        n_subgoals = jnp.repeat(subgoals[..., None, :], self.config["low_actor_num_samples"], axis=-2)
        n_actions = self.compute_flow_actions(n_observations, n_subgoals, n_noises)
        n_actions_q = self.network.select("low_critic")(n_observations, n_subgoals, n_actions).min(axis=0)
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
        ex_options = example_batch["high_subgoals"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["high_goals"]
        ex_times = ex_actions[..., :1]

        observation_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

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
            action_dim=observation_dim,
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
            high_value=(high_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_options)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goals, ex_options)),
            high_actor_bc_flow=(high_actor_bc_flow_def, (ex_observations, ex_goals, ex_options, ex_times)),
            low_value=(low_value_def, (ex_observations, ex_goals)),
            low_critic=(low_critic_def, (ex_observations, ex_goals, ex_actions)),
            target_low_critic=(copy.deepcopy(low_critic_def), (ex_observations, ex_goals, ex_actions)),
            low_actor_bc_flow=(low_actor_bc_flow_def, (ex_observations, ex_options, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
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
            agent_name="aso",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
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