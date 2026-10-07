import copy
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from qso.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from qso.utils.networks import (
    GCActorVectorField,
    GCActor,
    GCValue,
    GaussianBidirectionalGRUEncoder,
    GaussianPrior,
)

class OPALAgent(flax.struct.PyTreeNode):
    """
    OPAL agent with a flow-matching high-level policy over the option space, learned with a Beta-VAE.
    
    Training is either "pretrain" (option loss only, for rep_pretrain_steps steps), "finetune" (encoder and prior frozen; low-level BC finetuning and high-level offline RL), "joint" (everything trained simultaneously if rep_pretrain_steps is 0).
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    phase: str = nonpytree_field(default="joint")

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @staticmethod
    def make_tx(config, phase):
        """
        Return the optimiser for a training phase.
        """
        adam = optax.adam(learning_rate=config["lr"])
        frozen = optax.set_to_zero()
        transforms = dict(
            option=frozen if phase == "finetune" else adam,
            low=adam,
            high=frozen if phase == "pretrain" else adam,
            target=frozen,
        )
        module_groups = dict(
            modules_option_rep="option",
            modules_option_prior="option",
            modules_low_actor="low",
            modules_high_value="high",
            modules_high_critic="high",
            modules_high_actor_bc_flow="high",
            modules_target_high_critic="target",
            )
        return optax.multi_transform(transforms, lambda params: {k: module_groups[k] for k in params})

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

    def chunk_log_prob(
        self,
        batch,
        grad_params,
        options
        ):
        """
        Return the decoder log-likelihood of each action chunk.
        """
        options = jnp.broadcast_to(
            options[:, None, :], (*batch["option_obs_chunks"].shape[:-1], options.shape[-1])
        )
        action_dist = self.network.select("low_actor")(
            batch["option_obs_chunks"],
            options,
            params=grad_params
        )

        log_prob = (action_dist.log_prob(batch["option_action_chunks"]) * batch["option_valids"]).sum(-1)
        return log_prob, action_dist

    def option_loss(
        self,
        batch,
        grad_params,
        option_dist,
        options
        ):
        """
        Compute the KL-regularised option loss.
        """
        log_prob, action_dist = self.chunk_log_prob(batch, grad_params, options)
        priors = self.network.select("option_prior")(
            batch["observations"],
            params=grad_params
        )
        kl = option_dist.kl_divergence(priors)
        option_loss = -log_prob.mean() + self.config["rep_beta"] * kl.mean()

        valids = batch["option_valids"][..., None]

        return option_loss, {
            "option_loss": option_loss,
            "log_prob": log_prob.mean(),
            "kl": kl.mean(),
            "mse": (((action_dist.mode() - batch["option_action_chunks"]) ** 2) * valids).sum() / (valids.sum() * batch["option_action_chunks"].shape[-1]),
            "std": (action_dist.scale_diag * valids).sum() / (valids.sum() * batch["option_action_chunks"].shape[-1]),
            "z_std": option_dist.scale_diag.mean(),
        }

    def low_actor_loss(
        self,
        batch,
        grad_params,
        options
        ):
        """
        Compute the low-level actor loss (finetuning).
        """
        log_prob, _ = self.chunk_log_prob(batch, grad_params, jax.lax.stop_gradient(options))
        actor_loss = - self.config["low_actor_bc_coef"] * log_prob.mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_log_prob": log_prob.mean(),
        }

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
        """Compute the total loss for the current training phase."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, option_rng, high_actor_rng = jax.random.split(rng, 3)

        option_dist = self.network.select("option_rep")(
            batch["option_obs_chunks"],
            batch["option_action_chunks"],
            seq_lengths=batch["option_valids"].sum(-1),
            params=None if self.phase == "finetune" else grad_params, # do not backpropagate through GRU if pretraining is over.
        )
        options = option_dist.sample(seed=option_rng)

        # --- every loss is computed and logged in every phase (so metric names never change); the phase picks which are trained. ---
        option_loss, option_info = self.option_loss(batch, grad_params, option_dist, options)
        for k, v in option_info.items():
            info[f"option/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, options)
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params, jax.lax.stop_gradient(options))
        for k, v in high_value_info.items():
            info[f"high_value/{k}"] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params, jax.lax.stop_gradient(options))
        for k, v in high_critic_info.items():
            info[f"high_critic/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(
            batch, grad_params, jax.lax.stop_gradient(options), high_actor_rng
        )
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        if self.phase == "pretrain":
            loss = option_loss
        else:
            loss = (
                low_actor_loss
                + high_value_loss
                + high_critic_loss
                + high_actor_loss
            )
            if self.phase == "joint":
                loss = loss + option_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["value_tau"] + tp * (1 - self.config["value_tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "high_critic")

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    def end_pretraining(self):
        """
        Freeze the encoder and prior, and start low-level BC finetuning and high-level training (low-level keeps its Adam state, but high-level starts from step 1).
        """
        assert self.phase == "pretrain", f"end_pretraining called in phase {self.phase!r}"
        tx = self.make_tx(self.config, "finetune")
        opt_state = tx.init(self.network.params)
        opt_state = opt_state._replace(
            inner_states={**opt_state.inner_states, "low": self.network.opt_state.inner_states["low"]}
        )
        return self.replace(network=self.network.replace(tx=tx, opt_state=opt_state), phase="finetune")

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
        return options

    @jax.jit
    def sample_options(
        self,
        observations,
        goals,
        seed,
        temperature=1.0,
    ):
        """Sample options from the BC flow model, keeping the best of `num_samples` under the high-level critic."""

        # --- high-level rejection sampling. ---
        n_noises = jax.random.normal(seed, (*observations.shape[:-1], self.config["high_actor_num_samples"], self.config["rep_dim"]))
        n_observations = jnp.repeat(observations[..., None, :], self.config["high_actor_num_samples"], axis=-2)
        n_goals = jnp.repeat(goals[..., None, :], self.config["high_actor_num_samples"], axis=-2)
        n_options = self.compute_flow_options(n_observations, n_goals, n_noises)
        n_options_q = self.network.select("high_critic")(n_observations, n_goals, n_options).min(axis=0)
        high_indices = jnp.argmax(n_options_q, axis=-1)[..., None, None]
        return jnp.take_along_axis(n_options, high_indices, axis=-2).squeeze(-2)

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
        if options is None:
            options = self.sample_options(observations, goals, high_seed)

        # --- low-level decoder. ---
        low_dist = self.network.select("low_actor")(observations, options, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)
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
        ex_obs_chunks = example_batch["option_obs_chunks"]
        ex_action_chunks = example_batch["option_action_chunks"]
        ex_seq_lengths = example_batch["option_valids"].sum(-1)
        ex_options = jnp.zeros(shape=(1, config["rep_dim"]))

        observation_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        option_rep_def = GaussianBidirectionalGRUEncoder(
            hidden_dims=config["rep_hidden_dims"],
            latent_dim=config["rep_dim"],
            gru_hidden_dim=config["rep_gru_hidden_dim"],
            gru_num_layers=config["rep_gru_num_layers"],
        )
        option_prior_def = GaussianPrior(
            hidden_dims=config["rep_hidden_dims"],
            latent_dim=config["rep_dim"]
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
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            state_dependent_std=True, 
        )

        network_info = dict(
            option_rep=(option_rep_def, (ex_obs_chunks, ex_action_chunks, ex_seq_lengths)),
            option_prior=(option_prior_def, (ex_observations,)),
            high_value=(high_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_options)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, ex_goals, ex_options)),
            high_actor_bc_flow=(high_actor_bc_flow_def, (ex_observations, ex_goals, ex_options, ex_times)),
            low_actor=(low_actor_def, (ex_observations, ex_options)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        phase = "pretrain" if config["rep_pretrain_steps"] > 0 else "joint"
        network_tx = cls.make_tx(config, phase)
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_high_critic"] = params["modules_high_critic"]

        config["observation_dim"] = observation_dim
        config["action_dim"] = action_dim
        config["low_discount"] = 1 - 1 / config["subgoal_steps"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config), phase=phase)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # --- environment hyperparameters. ---
            observation_dim=ml_collections.config_dict.placeholder(int),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimensions (will be set automatically).
            # --- agent hyperparameters. ---
            agent_name="opal",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            # --- representation hyperparameters. ---
            rep_pretrain_steps=100000,  # Primitive-learning steps before the encoder and prior are frozen, counted within offline_train_steps (0 trains everything jointly).
            rep_dim=10,  # Option representation dimension, dim(Z).
            rep_hidden_dims=(512, 512),  # Hidden dimensions of the encoder's state MLP and of the prior.
            rep_gru_hidden_dim=512,  # Hidden state dimension of each direction of the OPAL encoder GRU.
            rep_gru_num_layers=2,  # Number of stacked bidirectional GRU layers in the OPAL encoder.
            rep_beta=0.1,  # KL penalty weight of the option loss.
            # --- value function hyperparameters. ---
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_tau=0.005,  # Target network update rate.
            value_expectile=0.7,  # IQL expectile.
            # --- actor hyperparameters. ---
            actor_hidden_dims=(512, 512, 512, 512),  # High-level actor network hidden dimensions.
            actor_flow_steps=6,  # Number of flow steps.
            awr_alpha=3.0,  # AWR temperature.
            awr_clip=100.0,  # AWR clip.
            low_actor_bc_coef=1.0,  # Weight of the low-level BC finetuning loss (0 for the option loss alone).
            high_actor_loss="awr", # High actor loss.
            high_actor_num_samples=32,  # "Best-of-n": options ranked under the critic.
            # --- dataset hyperparameters. ---
            dataset_class="HRGCDataset",  # Dataset class name.
            discount=0.99,  # Discount factor.
            high_env_reward_relabel=True,  # Whether to relabel the high-level reward with the environment success function instead of goal-index equality.
            high_p_curgoal=0.1,  # Probability of using the current state as the value goal.
            high_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            high_p_randomgoal=0.4,  # Probability of using a random state as the value goal.
            subgoal_steps=25,  # Actor subgoal steps.
            hierarchical_agent=False, # Whether goals need to be sampled for a low-level value function.
            low_env_reward_relabel=False,  # Whether to relabel the low-level reward with the environment success function instead of goal-index equality.
            low_p_curgoal=0.1,  # Probability of using the current state as the value goal.
            low_p_trajgoal=0.85,  # Probability of using a future state in the same trajectory as the value goal.
            low_p_randomgoal=0.05,  # Probability of using a random state as the value goal.
            low_discount=ml_collections.config_dict.placeholder(float) # set based on subgoal steps.
            )
    )
    return config
