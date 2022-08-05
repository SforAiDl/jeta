import random
from dataset import Dataset

class TaskDataset():
    
    def __init__(self, dataset: Dataset, num_ways: int, num_shots: int):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots

    def sample_random(self):
        i = random.randint(0, len(self.dataset) - 1)
        return self.dataset[i]

    def sample(self):
        labels = random.sample(self.dataset.labels, k=self.num_ways)
        taskset = []
        for label in labels:
            taskset.extend([self.dataset[i] for i in random.sample(self.dataset.labels_to_indices[label],k=self.num_shots)])
        return taskset