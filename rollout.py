import random

import torch
import numpy as np

from utils import one_hot

class FakeRollout:
    def __init__(self, rollout_count, noise_dim, space_dim, n_classes, tensor):
        self.noise = tensor(np.random.normal(0, 1, (rollout_count, noise_dim)))
        self.space = tensor(np.random.uniform(-1, 1, (len(self.noise), space_dim)))
        self.label = torch.tensor(np.random.randint(0, n_classes, len(self.noise)),
                dtype=torch.long).to(self.space.device)

        self._one_hot_label(n_classes)

        self.indices = None

    def batch(self, size):
        self.indices = self._get_indices(len(self.noise), size)
        return self.roll()

    def roll(self):
        return (
                self.noise[self.indices],
                self.space[self.indices],
                self.one_hot_label[self.indices],
                )

    def _one_hot_label(self, n_classes):
        self.one_hot_label = torch.zeros((self.label.shape[0], n_classes),
                dtype=self.space.dtype).to(self.space.device)

        self.one_hot_label[range(self.label.shape[0]), list(self.label)] = 1.

    def get_indices(self, size, batch_size):
        return self._get_indices(size, batch_size)
    def _get_indices(self, size, batch_size):
        return random.sample(range(size), batch_size)

    def label_cuts(self):
        return [ (0, len(self.label)) ]

