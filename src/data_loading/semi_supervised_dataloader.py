from typing import Optional

import numpy as np
from torch.utils.data import DataLoader


class SemiSupervisedDataLoader:
    """
        This class takes two data loaders, and returns a custom dataloader
        interleaving batches from the two data loaders (with a boolean indicating
        from which one it comes from). This is usefull for semi-supervised learning, 
        where we have to learn from supervised and unsupervised data.
    """

    def __init__(self, supervised_loader: DataLoader, unsupervised_loader: DataLoader, seed: Optional[int]=None):
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader

        self.bool_array = np.array([True] * len(supervised_loader) + [False] * len(unsupervised_loader))
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
    
    def __iter__(self):
        np.random.shuffle(self.bool_array)

        self.iter_supervised = iter(self.supervised_loader)
        self.iter_unsupervised = iter(self.unsupervised_loader)
        self.bool_array_index = 0

        return self

    def __next__(self):
        if self.bool_array_index >= len(self.bool_array):
            raise StopIteration

        is_supervised = self.bool_array[self.bool_array_index]
        self.bool_array_index += 1
        
        if is_supervised:
            return True, next(self.iter_supervised)
        else:
            return False, next(self.iter_unsupervised)

    def __len__(self):
        return len(self.bool_array)