import collections
import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """
    Return the size of the dataset.
    """
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


def get_terminal_locs(terminals):
    """
    Return the last index of each trajectory, treating a trailing partial trajectory as ending on the final row.
    """
    (terminal_locs,) = np.nonzero(terminals > 0)
    if len(terminal_locs) == 0 or terminal_locs[-1] != len(terminals) - 1:
        terminal_locs = np.concatenate([terminal_locs, [len(terminals) - 1]])
    return terminal_locs


def build_initial_idx(terminals):
    """
    Return an array whose i-th entry is the first index of the trajectory containing i.
    """
    terminal_locs = get_terminal_locs(terminals)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    lengths = terminal_locs - initial_locs + 1
    return np.repeat(initial_locs, lengths).astype(np.int32)


def build_final_idx(terminals):
    """
    Return an array whose i-th entry is the last index of the trajectory containing i.
    """
    terminal_locs = get_terminal_locs(terminals)
    lengths = np.diff(np.concatenate([[-1], terminal_locs]))
    return np.repeat(terminal_locs, lengths).astype(np.int32)


class Dataset(FrozenDict):
    """
    Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes "observations" is always present in the keys. If
    "next_observations" is not present, it will be inferred from "observations" by shifting the indices by 1. In this
    case, set "valids" appropriately to mask out the last state of each trajectory.
    """
    @classmethod
    def create(cls, freeze=True, **fields):
        """
        Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if "valids" in self._dict:
            (self.valid_idxs,) = np.nonzero(self["valids"] > 0)

        (self.terminal_locs,) = np.nonzero(self["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        self.initial_idx = build_initial_idx(self["terminals"])
        self.final_idx = build_final_idx(self["terminals"])

    def get_random_idxs(self, num_idxs):
        """
        Return `num_idxs` random indices.
        """
        if "valids" in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """
        Sample a batch of transitions.
        """
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        """
        Return a subset of the dataset given the indices.
        """
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if "next_observations" not in result:
            result["next_observations"] = self._dict["observations"][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """
    Class to extend Dataset to support adding transitions (e.g. for offline-to-online RL). 
    """
    @classmethod
    def create(cls, transition, size):
        """
        Creates a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """
        Creates a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = dataset.episode_start = get_size(init_dataset)
        dataset.initial_idx[: dataset.size] = build_initial_idx(init_dataset["terminals"])
        dataset.final_idx[: dataset.size] = build_final_idx(init_dataset["terminals"])
        terminal_locs = get_terminal_locs(init_dataset["terminals"])
        dataset.episodes = collections.deque(zip(np.concatenate([[0], terminal_locs[:-1] + 1]), terminal_locs))
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0
        self.episode_start = 0
        self.episodes = collections.deque()
        self.initial_idx = np.zeros(self.max_size, dtype=np.int32)
        self.final_idx = np.zeros(self.max_size, dtype=np.int32)

    def get_random_idxs(self, num_idxs):
        """
        Return `num_idxs` random indices.
        Invalid rows include the final row of each trajectory (which transitions nowhere), plus the trajectory currently being added to the replay buffer.
        """
        idxs = np.random.randint(self.size, size=num_idxs)

        if "valids" not in self._dict:
            return idxs

        invalid = self._dict["valids"][idxs] == 0
        if invalid.mean() > 0.5: # rejection sampling is cheaper if invalid rows are rare.
            (valid_idxs,) = np.nonzero(self._dict["valids"][: self.size] > 0)
            assert len(valid_idxs) > 0, "no complete trajectory in the buffer yet"
            return valid_idxs[np.random.randint(len(valid_idxs), size=num_idxs)]
        while invalid.any():
            idxs[invalid] = np.random.randint(self.size, size=int(invalid.sum()))
            invalid = self._dict["valids"][idxs] == 0

        return idxs

    def add_transition(self, transition):
        """
        Add a transition to the replay buffer.
        """
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        # --- remove oldest trajectory from the replay buffer. ---
        while self.episodes and (self.pointer - self.episodes[0][0]) % self.max_size <= (self.episodes[0][1] - self.episodes[0][0]) % self.max_size:
            start, end = self.episodes.popleft()
            self._dict["valids"][start : end + 1] = 0.0

        jax.tree_util.tree_map(set_idx, self._dict, transition)

        # --- make transitions from the trajectory currently being rolled out invalid to avoid being sampled. ---
        self._dict["valids"][self.pointer] = 0.0
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def record_trajectory(self, start, end):
        """
        Record a completed trajectory, making its rows sampleable (only keep the final row as invalid). 
        """
        self.initial_idx[start : end + 1] = start
        self.final_idx[start : end + 1] = end
        self._dict["valids"][start:end] = 1.0
        self._dict["valids"][end] = 0.0
        self.episodes.append((start, end))

    def end_episode(self):
        """
        Record trajectory currently being sampled, filling in its final_idx and make its rows sampleable.
        """
        if self.pointer == self.episode_start:
            return
        end = (self.pointer - 1) % self.max_size

        if end < self.episode_start: # record any trajectory that wrapped past the end of the buffer as two separate trajectories. 
            self.record_trajectory(self.episode_start, self.max_size - 1)
            self.record_trajectory(0, end)
        else:
            self.record_trajectory(self.episode_start, end)

        self.episode_start = self.pointer

    def clear(self):
        """
        Clear the replay buffer.
        """
        self.size = self.pointer = self.episode_start = 0
        self.episodes.clear()


@dataclasses.dataclass
class GCDataset(Dataset):
    """
    Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - p_curgoal: Probability of using the current state as the goal.
    - p_trajgoal: Probability of using a future state in the same trajectory as the goal.
    - p_randomgoal: Probability of using a random state as the goal.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        env: Environment required for compute_success. 
        env_name: Environment name required for compute_success. 
    """
    dataset: Dataset
    config: Any
    env: Any
    env_name: Any

    def __post_init__(self):
        assert np.isclose(
            self.config["p_curgoal"] + self.config["p_trajgoal"] + self.config["p_randomgoal"], 1.0
        )
        self.compute_success = (
            get_compute_success(self.env, self.env_name) if self.config["env_reward_relabel"] else None
        )

    @property
    def size(self):
        """
        Number of transitions currently in the underlying dataset.
        """
        return self.dataset.size

    def sample(self, 
        batch_size: int, 
        idxs=None, 
        evaluation=False
        ):
        """
        Sample a batch of transitions with goals from the dataset. 
        Stored in the keys "goals". 
        It also computes the "rewards" and "masks" based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        goal_idxs = self.sample_goals(
            idxs,
            self.config["p_curgoal"],
            self.config["p_trajgoal"],
            self.config["p_randomgoal"],
        )
        batch["goals"] = self.get_observations(goal_idxs)
        if self.config["env_reward_relabel"] == True:
            successes = self.compute_success(batch["observations"], batch["goals"]).astype(float)
        else:
            successes = (idxs == goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - 1.0

        return batch

    def sample_goals(
        self, 
        idxs, 
        p_curgoal, 
        p_trajgoal, 
        p_randomgoal
        ):
        """
        Sample goals for the given indices.
        """
        batch_size = len(idxs)

        # --- random goals. --- 
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # --- goals uniformly sampled from the same trajectory (excluding the current state, unless it is the final state). --- 
        final_state_idxs = self.dataset.final_idx[idxs]
        distances = np.random.rand(batch_size)  # in [0, 1)
        traj_goal_idxs = np.round(
            (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
        ).astype(int)
        
        # --- goals at the current state. --- 
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_observations(
        self, 
        idxs
        ):
        """
        Return the observations for the given indices.
        """
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset["observations"])


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """
    Dataset class for hierarchical goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - p_curgoal: Probability of using the current state as the goal.
    - p_trajgoal: Probability of using a future state in the same trajectory as the goal.
    - p_randomgoal: Probability of using a random state as the goal.
    - subgoal_steps: Number of subgoal steps
    """
    def compute_subgoal_idxs(
        self, 
        idxs, 
        final_state_idxs, 
        high_goal_idxs, 
        subgoal_steps
        ):
        """
        Compute the next indices for high-level goals.
        """
        batch_size = len(idxs)
        subgoal_steps = np.full(batch_size, subgoal_steps)

        # --- clip to the end of the trajectory. --- 
        subgoal_steps = np.minimum(subgoal_steps, final_state_idxs - idxs)

        # --- clip to the high-level goal (diff_idxs might be < idxs if the goal is randomly drawn). --- 
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < subgoal_steps)
        subgoal_steps = np.where(should_clip, diff_idxs, subgoal_steps)

        return idxs + subgoal_steps, subgoal_steps

    def sample(
        self, 
        batch_size: int, 
        idxs=None, 
        evaluation=False
        ):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.dataset.final_idx[idxs]

        # --- sample high-level goals. ---
        high_goal_idxs = self.sample_goals(
            idxs,
            self.config["p_curgoal"],
            self.config["p_trajgoal"],
            self.config["p_randomgoal"],
        )
        high_subgoal_idxs, batch["subgoal_steps"] = self.compute_subgoal_idxs(
            idxs,
            final_state_idxs,
            high_goal_idxs,
            self.config["subgoal_steps"],
        )
        batch["goals"] = self.get_observations(high_goal_idxs)
        batch["subgoals"] = self.get_observations(high_subgoal_idxs)
        if self.config["env_reward_relabel"] == True:
            successes = self.compute_success(batch["observations"], batch["goals"]).astype(float)
        else:
            successes = (idxs == high_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - 1.0

        return batch
    

@dataclasses.dataclass
class HRGCDataset(HGCDataset):
    """
    Dataset class for horizon-reduction goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - high_p_curgoal: Probability of using the current state as the goal.
    - high_p_trajgoal: Probability of using a future state in the same trajectory as the goal.
    - high_p_randomgoal: Probability of using a random state as the goal.
    - subgoal_steps: Number of subgoal steps
    - hierarchical_agent: Whether low level goals should also be sampled (not required if a flat-AC agent is used)
    - low_p_curgoal: Probability of using the current state as the goal.
    - low_p_trajgoal: Probability of using a future state in the same trajectory as the goal.
    - low_p_randomgoal: Probability of using a random state as the goal.
    """
    def __post_init__(self):
        for level in ["high", "low"]:
            assert np.isclose(
                self.config[f"{level}_p_curgoal"]
                + self.config[f"{level}_p_trajgoal"]
                + self.config[f"{level}_p_randomgoal"],
                1.0,
            )
        relabel = self.config["high_env_reward_relabel"] or self.config["low_env_reward_relabel"]
        self.compute_success = get_compute_success(self.env, self.env_name) if relabel else None

    def sample(
        self, 
        batch_size: int, 
        idxs=None, 
        evaluation=False
        ):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.dataset.final_idx[idxs]

        # --- sample high-level goals. ---
        high_goal_idxs = self.sample_goals(
            idxs,
            self.config["high_p_curgoal"],
            self.config["high_p_trajgoal"],
            self.config["high_p_randomgoal"],
        )
        batch["high_goals"] = self.get_observations(high_goal_idxs)
        obs_chunk_idxs = np.minimum(idxs[:, None] + np.arange(self.config["subgoal_steps"] + 1)[None, :], final_state_idxs[:, None]) # window spans the whole option (including its final state) so a goal sitting exactly at the option end counts as reached.

        if self.config["high_env_reward_relabel"] == True:
            step_successes = self.compute_success(self.get_observations(obs_chunk_idxs), batch["high_goals"][:, None, :]).astype(bool)
        else:
            step_successes = obs_chunk_idxs == high_goal_idxs[:, None]

        reached = step_successes.any(axis=-1)
        high_successes = reached.astype(float)
        first_success = np.where(reached, step_successes.argmax(axis=-1), self.config["subgoal_steps"])
        batch["high_masks"] = 1.0 - high_successes
        batch["high_subgoal_steps"] = np.clip(first_success, 1, final_state_idxs - idxs) # option lasts at least one step, even where the current state already reaches the goal.
        batch["high_subgoals"] = self.get_observations(idxs + batch["high_subgoal_steps"])

        low_eff_discount = self.config["discount"] ** (1 / self.config["subgoal_steps"])
        batch["high_discounts_real"] = low_eff_discount ** batch["high_subgoal_steps"]
        batch["high_rewards_real"] = - (1 - batch["high_discounts_real"]) / (1 - self.config["discount"]) # option reward accumulation is scaled so that a full-length option costs exactly -1. 
        batch["high_discounts_synthetic"] = np.full_like(batch["high_rewards_real"], self.config["discount"])
        batch["high_rewards_synthetic"] = batch["high_rewards_real"] * high_successes - 1.0 * (1 - high_successes)

        batch["option_start"] = batch["observations"]
        batch["option_end"] = self.get_observations(np.minimum(idxs + self.config["subgoal_steps"], final_state_idxs)) # Option always looks 25 steps ahead, even if the goal is reached within the option.

        # --- sample low-level goals (if requested). --- 
        if self.config["hierarchical_agent"]:
            low_goal_idxs = self.sample_low_goals(
                idxs,
                self.config["low_p_curgoal"],
                self.config["low_p_trajgoal"],
                self.config["low_p_randomgoal"],
                )
            batch["low_goals"] = self.get_observations(low_goal_idxs)

            if self.config["low_env_reward_relabel"] == True:
                successes = self.compute_success(batch["observations"], batch["low_goals"]).astype(float)
            else:
                successes = (idxs == low_goal_idxs).astype(float)

            batch["low_masks"] = 1.0 - successes
            batch["low_rewards"] = successes - 1.0

        if self.config["agent_name"] == "qc": # goal-dependent chunks for QC. 
            batch["synthetics"] = np.arange(self.config["subgoal_steps"])[None, :] >= batch["high_subgoal_steps"][:, None]
            batch["action_chunks"] = np.where(~batch["synthetics"][..., None], self.get_chunks(idxs, "actions"), 0.0) 

        if self.config["agent_name"] == "opal": # goal-independent chunks for OPAL. 
            batch["option_valids"] = idxs[:, None] + np.arange(self.config["subgoal_steps"])[None, :] <= final_state_idxs[:, None] 
            batch["option_obs_chunks"] = self.get_chunks(idxs, "observations")
            batch["option_action_chunks"] = np.where(batch["option_valids"][..., None], self.get_chunks(idxs, "actions"), 0.0)

        return batch
    
    def sample_goals(
        self, 
        idxs, 
        p_curgoal, 
        p_trajgoal, 
        p_randomgoal
        ):
        """
        Sample high level goals for the given indices.
        """
        batch_size = len(idxs)

        # --- random goals. --- 
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # --- goals uniformly sampled from the same trajectory following the option (unless it is the final state). --- 
        final_state_idxs = self.dataset.final_idx[idxs]
        distances = np.random.rand(batch_size)  # in [0, 1)
        traj_goal_idxs = np.minimum(np.round(
            (idxs + self.config["subgoal_steps"]) * distances + final_state_idxs * (1 - distances)), 
            final_state_idxs).astype(int)

        # --- goals uniformly sampled from within the option (excluding the current state, unless it is the final state). ---
        distances = np.random.rand(batch_size)  # in [0, 1)
        option_goal_idxs = np.minimum(np.round(
            (idxs + 1) * distances + (idxs + self.config["subgoal_steps"]) * (1 - distances)),
            final_state_idxs).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = option_goal_idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, option_goal_idxs, goal_idxs)

        return goal_idxs

    def sample_low_goals(
        self, 
        idxs, 
        p_curgoal, 
        p_trajgoal, 
        p_randomgoal, 
        ):
        """
        Sample low level goals for the given indices using geometric sampling. 
        """
        batch_size = len(idxs)

        # --- random goals. --- 
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # --- goals geometrically sampled from the same trajectory (excluding the current state, unless it is the final state). ---
        final_state_idxs = self.dataset.final_idx[idxs]
        low_discount = 1 - 1 / self.config["subgoal_steps"]
        offsets = np.random.geometric(p= 1 - low_discount, size=batch_size)  # in [1, inf)
        traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        
        # --- goals at the current state. --- 
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs
    
    def get_chunks(
        self,
        idxs,
        key
        ):
        """
        Return the length-`subgoal_steps` sequences starting at the given indices.
        """
        chunk_idxs = np.minimum(
            idxs[:, None] + np.arange(self.config["subgoal_steps"])[None, :], self.dataset.final_idx[idxs][:, None]
        )
        return jax.tree_util.tree_map(lambda arr: arr[chunk_idxs], self.dataset[key])


def _object_component_cols(object_len, num_objects, stride, num_used, offset=0):
    """
    One column-index array per component of an object's state.
    Args:
        object_len: Width of object block at tail of observation.
        num_objects: Number of objects in the block.
        stride: Width of single object's state.
        num_used: Number of leading components of each object's state to compare.
        offset: Start of this group of objects within the object block.
    """
    base = offset + np.arange(num_objects) * stride - object_len
    return [base + j for j in range(num_used)]


def _component_norm(observation, goal, component_cols, scaler=None):
    """
    Per-object Euclidean distance.
    """
    total = None
    for cols in component_cols:
        ob_component = observation[..., cols]
        goal_component = goal[..., cols]
        if scaler is not None:
            ob_component = ob_component / scaler
            goal_component = goal_component / scaler
        d = ob_component - goal_component
        total = d * d if total is None else total + d * d
    return np.sqrt(total)


def get_compute_success(env, env_name):
    """
    Returns a function to compute success from observations.
    """
    assert env.unwrapped._ob_type != 'pixels', "compute_success is not supported for pixel observations"

    if "maze" in env_name:
        _goal_tol = env.unwrapped._goal_tol
        # --- the agent xy is at the head of the observation, so the columns are plain. ---
        _xy_cols = [0, 1]

        def _compute_success(observation, goal):
            # --- extract relevant states ---
            return _component_norm(observation, goal, _xy_cols) <= _goal_tol

    elif "cube" in env_name:
        _num_cubes = env.unwrapped._num_cubes
        # --- hardcoded in environment, therefore also hardcoded here ---
        _xyz_scaler = 10.0

        # --- object (cube) states live at the tail of the observation, after the arm state ---
        object_len = _num_cubes * 9
        _xyz_cols = _object_component_cols(object_len, _num_cubes, 9, 3)

        def _compute_success(observation, goal):
            # --- extract relevant states ---
            cube_successes = _component_norm(observation, goal, _xyz_cols, _xyz_scaler) <= 0.04

            # --- compute logical AND for all cubes ---
            return cube_successes.all(axis=-1)

    elif "puzzle" in env_name:
        _num_buttons = env.unwrapped._num_buttons
        # --- hardcoded in environment, therefore also hardcoded here ---
        _num_button_states = 2

        # --- object (button) states live at the tail of the observation, after the arm state ---
        object_len = _num_buttons * 4
        _button_cols = _object_component_cols(object_len, _num_buttons, 4, _num_button_states)

        def _compute_success(observation, goal):
            # --- extract relevant states ---
            button_successes = _component_norm(observation, goal, _button_cols) < 0.5

            # --- compute logical AND for all buttons ---
            return button_successes.all(axis=-1)

    elif "scene" in env_name:
        _num_cubes = env.unwrapped._num_cubes
        _num_buttons = env.unwrapped._num_buttons
        # --- hardcoded in environment, therefore also hardcoded here. ---
        _num_button_states = 2
        _xyz_scaler = 10.0
        _drawer_scaler = 18.0
        _window_scaler = 15.0

        # --- for extracting object states from observation and goal. ---
        object_len = (_num_cubes * 9) + (_num_buttons * 4) + 2 + 2
        cube_end = _num_cubes * 9
        btn_end = cube_end + (_num_buttons * 4)
        _xyz_cols = _object_component_cols(object_len, _num_cubes, 9, 3)
        _button_cols = _object_component_cols(object_len, _num_buttons, 4, _num_button_states, offset=cube_end)
        # --- the drawer and the window are single scalars, indexed without a trailing axis. ---
        _drawer_col = btn_end - object_len
        _window_col = btn_end + 2 - object_len

        def _compute_success(observation, goal):
            # --- extract relevant states. ---
            drawer_pos = observation[..., _drawer_col] / _drawer_scaler
            goal_drawer_pos = goal[..., _drawer_col] / _drawer_scaler
            window_pos = observation[..., _window_col] / _window_scaler
            goal_window_pos = goal[..., _window_col] / _window_scaler

            cube_successes = _component_norm(observation, goal, _xyz_cols, _xyz_scaler) <= 0.04
            button_successes = _component_norm(observation, goal, _button_cols) < 0.5
            drawer_successes = np.abs(drawer_pos - goal_drawer_pos) <= 0.04
            window_successes = np.abs(window_pos - goal_window_pos) <= 0.04

            # --- compute logical AND for all cubes, buttons, the drawer and the window. ---
            return cube_successes.all(axis=-1) & button_successes.all(axis=-1) & drawer_successes & window_successes

    else:
        raise ValueError(f"Environment {env_name} not supported")

    return _compute_success