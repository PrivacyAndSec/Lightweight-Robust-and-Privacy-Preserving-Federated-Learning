import os
from torchvision.transforms import ToTensor
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Subset, Dataset, DataLoader, random_split, sampler
import random


def niid_idx(dataset) -> [int]:
    # delete the test data with target label
    id_idxs = []
    for i in range(0, 10):
        id_idxs.append([])
    for idx, (image, label) in enumerate(dataset):
        id_idxs[label].append(idx)

    return id_idxs



class DatasetSource:
    def __init__(self, dataset_name):
        if dataset_name == "mnist":
            self.train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=ToTensor())
            self.test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=ToTensor())
            self.n = 28
            self.m = 28
        elif dataset_name == "cifar10":
            self.train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=ToTensor())
            self.test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=ToTensor())
            self.n = 32
            self.m = 32
        elif dataset_name == "fmnist":
            self.train_dataset = datasets.FashionMNIST('data/fmnist', train=True, download=True, transform=ToTensor())
            self.test_dataset = datasets.FashionMNIST('data/fmnist', train=False, download=True, transform=ToTensor())
            self.n = 28
            self.m = 28

    def get_train_dataloader(self, arg, batch_size=64):
        self.ci_dataloader = []
        num_items = int(len(self.train_dataset) / arg.N * 2)
        for i in range(0, arg.N):
            idxs = iid_sampling(self.train_dataset, num_items)
            ci_dataset = Subset(self.train_dataset, list(idxs))
            self.ci_dataloader.append(DataLoader(ci_dataset, batch_size=batch_size, shuffle=True))


        return self.ci_dataloader



    def get_test_dataloader(self, batch_size=64):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)


def niid_sampling(i_idxs, num_items):
    return np.random.choice(i_idxs, num_items, replace=True)

def iid_sampling(dataset, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_items:
    :return: dict of image index
    """
    all_idxs = [i for i in range(len(dataset))]
    return set(np.random.choice(all_idxs, num_items, replace=True))
