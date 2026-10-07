from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Helper function to split the random number generator key before each call to the function.
    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """
    Flatten a dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """
    Append values to the corresponding lists in the dictionary.
    """
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    action_dim,
    task_id=None,
    num_eval_episodes=50,
    eval_temperature=0,
    eval_gaussian=None,
):
    """
    Evaluate the agent in the environment.

    Args:
        agent: Agent.
        action_dim: Environment action dimension, used to unfold a chunked action into single steps.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics and trajectories.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    option_fn = supply_rng(agent.sample_options, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))) if hasattr(agent, "sample_options") else None
    trajs = []
    stats = defaultdict(list)

    for i in trange(num_eval_episodes):
        traj = defaultdict(list)

        observation, info = env.reset(options=dict(task_id=task_id))
        goal = info.get('goal')
        done = False
        step = 0
        action_queue = []
        options = None

        while not done:

            # --- for OPAL, commit to each option for `subgoal_steps` steps. ---
            action_kwargs = {}
            if option_fn is not None:
                if step % agent.config["subgoal_steps"] == 0:
                    options = option_fn(observations=observation, goals=goal)
                action_kwargs["options"] = options

            # --- for chunked policies, execute all actions in the sequence. ---
            if len(action_queue) == 0:
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature, **action_kwargs)
                action_queue.extend(np.array(action).reshape(-1, action_dim))
            action = action_queue.pop(0)

            # --- for Gaussian policies. --- 
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            # --- step environment. --- 
            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            # --- save transition. ---
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs
