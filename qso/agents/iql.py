import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from qso.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from qso.utils.networks import GCActor, GCValue


class IQLAgent(flax.struct.PyTreeNode):
    """
    Implicit Q-learning (IQL) agent.
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
        Compute the IQL value loss.
        """

        q1, q2 = self.network.select("target_critic")(
            batch["observations"], batch["goals"], batch["actions"]
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(
            batch["observations"], batch["goals"], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config["value_expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """
        Compute the IQL critic loss.
        """

        next_v = self.network.select("value")(
            batch["next_observations"], batch["goals"]
        )
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["goals"],
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

    def actor_loss(self, batch, grad_params, rng=None):
        """
        Compute the actor loss (AWR or DDPG+BC).
        """

        if self.config["actor_loss"] == "awr":

            v = self.network.select("value")(
                batch["observations"], batch["goals"]
            )
            q1, q2 = self.network.select("critic")(
                batch["observations"], batch["goals"], batch["actions"]
            )
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config["awr_alpha"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select("actor")(
                batch["observations"], batch["goals"], params=grad_params
            )
            log_prob = dist.log_prob(batch["actions"])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
            }
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )

            return actor_loss, actor_info

        elif self.config["actor_loss"] == "ddpgbc":
            dist = self.network.select("actor")(
                batch["observations"], batch["goals"], params=grad_params
            )
            q_actions = jnp.clip(dist.mode(), -1, 1)
            q1, q2 = self.network.select("critic")(
                batch["observations"], batch["goals"], q_actions
            )
            q = jnp.minimum(q1, q2)

            # --- normalise q values by the absolute mean to make the loss scale invariant ---
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])

            bc_loss = -(self.config["ddpgbc_alpha"] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f"Unsupported actor loss: {self.config['actor_loss']}")

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """
        Compute the total loss.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
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
        dist = self.network.select("actor")(
            observations, goals, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_batch["observations"]
        ex_observations = ex_batch["observations"]
        ex_actions = ex_batch["actions"]

        observation_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        # --- define networks ---
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=1,
        )
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            num_ensembles=2,
        )
        actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions),),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

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
            agent_name="iql",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            # --- value function hyperparameters. ---
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_tau=0.005,  # Target network update rate.
            value_expectile=0.7,  # IQL expectile.
            # --- actor hyperparameters. ---
            actor_loss="ddpgbc",  # Actor loss type ("awr" or "ddpgbc").
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            awr_alpha=3.0,  # AWR temperature.
            ddpgbc_alpha=0.3,  # DDPG+BC temperature.
            # --- dataset hyperparameters. ---
            dataset_class="GCDataset",  # Dataset class name.
            discount=0.995,  # Discount factor.
            env_reward_relabel=False, # Whether to use lenient reward relabelling. 
            p_curgoal=0.1,  # Probability of using the current state as the value goal.
            p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            p_randomgoal=0.4,  # Probability of using a random state as the value goal.
        )
    )
    return config
