from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.data_loading.constants import *
from src.data_loading.data_transforms import default_transform
from src.data_loading.collate_fn import *
from src.data_loading.semi_supervised_dataloader import SemiSupervisedDataLoader

def get_data_loaders(dataset_name: str, 
                     p_test: float, 
                     p_val: float, 
                     p_supervised: float,
                     batch_size: int,
                     num_workers: int=0,
                     shuffle: bool=True, 
                     seed: Optional[int]=None
                     ):
    """
        get_data_loaders takes as input the name of a dataset,
        the proportion of testing data, 
        the proportion of validation data within the remaining data,
        the proportion of supervised data within the remaining data,
        the batch size and number of workers used for the data loaders,
        and return the different loaders :
        test, validation, supervised, unsupervised, semi-supervised
        and the image shape.
    """


    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=default_transform)
        img_shape = MNIST_IMG_SHAPE
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)
        img_shape = CIFAR10_IMG_SHAPE
    else:
        raise ValueError("Unknown dataset:", str(dataset_name))
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Computing sizes
    initial_size = len(dataset)

    test_size = int(p_test * initial_size)
    res_size = initial_size - test_size

    val_size = int(p_val * res_size)
    train_size = res_size - val_size

    supervised_size = int(p_supervised * train_size)
    unsupervised_size = train_size - supervised_size

    assert initial_size == test_size + val_size + supervised_size + unsupervised_size, "Error computing sizes."

    # Recovering datasets
    res_dataset, test_dataset = random_split(dataset=dataset, lengths=[res_size, test_size])
    train_dataset, val_dataset = random_split(dataset=res_dataset, lengths=[train_size, val_size])
    supervised_dataset, unsupervised_dataset = random_split(dataset=train_dataset, lengths=[supervised_size, unsupervised_size])

    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             num_workers=num_workers, 
                             collate_fn=lambda batch : jax_supervised_collate_fn(batch, img_shape)
                             )
    val_loader = DataLoader(val_dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             num_workers=num_workers, 
                             collate_fn=lambda batch : jax_supervised_collate_fn(batch, img_shape)
                             )
    supervised_loader = DataLoader(supervised_dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             num_workers=num_workers, 
                             collate_fn=lambda batch : jax_supervised_collate_fn(batch, img_shape)
                             )
    unsupervised_loader = DataLoader(unsupervised_dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             num_workers=num_workers, 
                             collate_fn=lambda batch : jax_unsupervised_collate_fn(batch, img_shape)
                             )
    
    semi_supervised_loader = SemiSupervisedDataLoader(supervised_loader=supervised_loader, 
                                                      unsupervised_loader=unsupervised_loader, 
                                                      seed=seed)

    res = {
        "test": test_loader,
        "validation": val_loader,
        "supervised": supervised_loader,
        "unsupervised": unsupervised_loader,
        "semi_supervised": semi_supervised_loader
    }
    print("Successfully loaded", dataset_name, "dataset.")
    print("Total num samples", initial_size)
    print("Num test samples:", test_size)
    print("Num validation samples:", val_size)
    print("Num supervised samples:", supervised_size)
    print("Num unsupervised samples:", unsupervised_size)
    return img_shape, res