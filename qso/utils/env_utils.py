import collections
import os
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from tqdm import tqdm

import ogbench
from qso.utils.datasets import Dataset

def make_env_and_datasets(dataset_name, dataset_dir=None, env_only=False):
    """
    Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        dataset_dir: Dataset directory.
        env_only: Whether to return the environment only.

    Returns:
        The environment if `env_only`, else a tuple of the environment, training dataset,
        validation dataset.
    """
    if env_only:
        env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
        return env

    else:
        # -- use compact dataset to save memory. --- 
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_dir=dataset_dir, compact_dataset=True
        )
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)

        # --- clip dataset actions. --- 
        eps = 1e-5
        train_dataset = train_dataset.copy(
            add_or_replace=dict(
                actions=np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
            )
        )
        val_dataset = val_dataset.copy(
            add_or_replace=dict(
                actions=np.clip(val_dataset["actions"], -1 + eps, 1 - eps)
            )
        )

        env.reset()
        return env, train_dataset, val_dataset
