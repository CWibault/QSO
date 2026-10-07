import math
from typing import Any, Dict, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import distrax


def default_init(scale=1.0):
    """
    Default kernel initialiser.
    """
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Ensemblise a module.
    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


def length_normalize(x):
    """
    Normalise the input along the last dimension to have a length of sqrt(dim).
    """
    return (
        x
        / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
        * jnp.sqrt(x.shape[-1])
    )


class LengthNormalize(nn.Module):
    """
    Layer to normalise the input along the last dimension to have a length of sqrt(dim).
    """
    @nn.compact
    def __call__(self, x):
        return (
            x
            / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
            * jnp.sqrt(x.shape[-1])
        )


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activate_final: Whether to apply activation to the final layer.
        layer_norm: Whether to apply layer normalization.
    """
    hidden_dims: Sequence[int]
    activate_final: bool = False
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.gelu(x)
                if self.layer_norm: 
                    x = nn.LayerNorm()(x)
        return x


class GCValue(nn.Module):
    """
    Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (set to None for scalar output).
        num_ensembles: Number of ensemble components.
    """
    hidden_dims: Sequence[int]
    output_dim: int = None
    num_ensembles: int = 2

    def setup(self):
        if self.num_ensembles > 1:
            mlp = ensemblize(MLP, self.num_ensembles)
        else:
            mlp = MLP
        output_dim = self.output_dim if self.output_dim is not None else 1
        value_net = mlp(
            (*self.hidden_dims, output_dim),
            activate_final=False,
        )
        self.value_net = value_net

    def __call__(
        self, 
        observations, 
        goals=None, 
        actions=None
        ):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        inputs = [observations]

        if goals is not None:
            inputs.append(goals)

        if actions is not None:
            inputs.append(actions)

        inputs = jnp.concatenate(inputs, axis=-1)
        v = self.value_net(inputs)

        if self.output_dim is None:
            v = v.squeeze(-1)

        return v


class GCActor(nn.Module):
    """
    Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        state_dependent_std: Whether to use state-dependent standard deviation.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = False

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))

    def __call__(
        self,
        observations,
        goals=None,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals.
            temperature: Scaling factor for the standard deviation.
        """
        outputs = self.actor_net(
            jnp.concatenate([observations, goals], axis=-1)
            )

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = jnp.clip(self.log_std_net(outputs), -5.0, 2.0)
        else:
            log_stds = jnp.zeros_like(means)

        return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
    

class GCActorVectorField(nn.Module):
    """
    Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
    """
    hidden_dims: Sequence[int]
    action_dim: int

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
        )

    @nn.compact
    def __call__(
        self, 
        observations, 
        goals, 
        actions, 
        times=None
        ):
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions.
            times: Times (optional).
        """
        if times is None:
            inputs = jnp.concatenate([observations, goals, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, goals, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class GaussianBidirectionalGRUEncoder(nn.Module):
    """
    Gaussian bidirectional GRU Encoder (OPAL).

    Following the paper, states pass through an MLP, are concatenated with actions and run through a stack of bidirectional GRUs;
    the final forward and backward hidden states of the top layer are projected to the mean and log std of z.

    Attributes:
        hidden_dims: Hidden layer dimensions of state MLP.
        gru_hidden_dim: Hidden dimension of each GRU direction.
        gru_num_layers: Number of stacked bidirectional GRU layers.
        latent_dim: Latent dimension.
    """

    hidden_dims: Sequence[int]
    gru_hidden_dim: int
    gru_num_layers: int
    latent_dim: int

    def setup(self):
        self.state_net = MLP(self.hidden_dims, activate_final=True)
        self.encoder_nets = [
            nn.Bidirectional(
                forward_rnn=nn.RNN(nn.GRUCell(features=self.gru_hidden_dim)),
                backward_rnn=nn.RNN(nn.GRUCell(features=self.gru_hidden_dim)),
                return_carry=True,
            )
            for _ in range(self.gru_num_layers)
        ]
        self.mean_net = nn.Dense(self.latent_dim, kernel_init=default_init(1e-2))
        self.log_std_net = nn.Dense(self.latent_dim, kernel_init=default_init(1e-2)) # Always learn. 

    def __call__(
        self,
        observation_chunks,
        action_chunks,
        seq_lengths=None,
        temperature=1.0,
    ):
        """
        Returns the option posterior distribution q(z | tau).

        Args:
            observation_chunks: Observation sequences.
            action_chunks: Action sequences.
            seq_lengths: Number of valid (non-padded) steps in each sequence.
            temperature: Scaling factor for the standard deviation.
        """
        outputs = jnp.concatenate([self.state_net(observation_chunks), action_chunks], axis=-1)
        # --- with seq_lengths, the forward carry is taken at the last valid step and the backward pass skips the padding. ---
        for encoder_net in self.encoder_nets:
            (h_forward, h_backward), outputs = encoder_net(outputs, seq_lengths=seq_lengths)
        outputs = jnp.concatenate([h_forward, h_backward], axis=-1)

        means = self.mean_net(outputs)
        log_stds = jnp.clip(self.log_std_net(outputs), -5.0, 2.0)

        return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)


class GaussianPrior(nn.Module):
    """
    Gaussian prior.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
    """

    hidden_dims: Sequence[int]
    latent_dim: int

    def setup(self):
        self.prior_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.latent_dim, kernel_init=default_init(1e-2))
        self.log_std_net = nn.Dense(self.latent_dim, kernel_init=default_init(1e-2))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return the prior distribution.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        outputs = self.prior_net(observations)
        means = self.mean_net(outputs)
        log_stds = jnp.clip(self.log_std_net(outputs), -5.0, 2.0)

        return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)