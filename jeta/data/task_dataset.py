import random
from typing import List, Tuple

import jax.numpy as jnp

from jeta.data.base_dataset import BaseDataset


class TaskDataset:
    """Generate tasks from JetaDataset

    Args:
        dataset(JetaDataset): dataset for geenrating tasks
        num_ways(int): number of labels to sample from
        num_shots(int): number of samples from each label
    """

    def __init__(self, dataset: BaseDataset, num_ways: int, num_shots: int):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots

    def sample_one(self) -> Tuple[jnp.ndarray, int]:
        """Randomly samples a task from the dataset

        Returns:
            Tuple[jnp.ndarray, int]: (data, target)
        """
        i = random.randint(0, len(self.dataset) - 1)
        X = jnp.array(self.dataset[i][0], dtype=jnp.float32)
        y = self.dataset[i][1]
        return X, y

    def sample(self) -> List[Tuple[jnp.ndarray, int]]:
        """Samples tasks from the dataset in NWays and KShots

        Returns:
            List[Tuple[jnp.ndarray, int]]: [(data, target),]
        """
        labels = random.sample(self.dataset.labels, k=self.num_ways)
        taskset = []
        for label in labels:
            taskset.extend(
                [
                    (
                        jnp.array(self.dataset[i][0], dtype=jnp.float32),
                        self.dataset[i][1],
                    )
                    for i in random.sample(
                        self.dataset.labels_to_indices[label], k=self.num_shots
                    )
                ]
            )
        return taskset

    def __iter__(self):
        for task in self.sample():
            yield task[0], task[1]
