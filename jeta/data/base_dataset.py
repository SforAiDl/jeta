from collections import defaultdict

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Wrapper for pytorch dataset with indices for task sampling

    Adapted from https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_dataset.pyx
    """

    def __init__(self, dataset):
        self.dataset = dataset
        labels_to_indices = defaultdict(list)
        indices_to_labels = defaultdict(int)
        for i in range(len(self.dataset)):
            try:
                label = self.dataset[i][1]
                if hasattr(label, "item"):
                    label = self.dataset[i][1].item()
            except ValueError as e:
                raise ValueError("Requires scalar labels. \n" + str(e))

            labels_to_indices[label].append(i)
            indices_to_labels[i] = label

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

        self._bookkeeping = {
            "labels_to_indices": self.labels_to_indices,
            "indices_to_labels": self.indices_to_labels,
            "labels": self.labels,
        }

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
