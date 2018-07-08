from torchvision import datasets
import torch
import numpy as np


class SparseMNIST(datasets.MNIST):
    def __init__(self, root, keep_percent, train=True, transform=None, target_transform=None, download=False, ign=-100,
                 showAll=True):
        """
        Replaces part of the original MNIST labels with an ignore label

        :param root: (string) Root directory of dataset where ``processed/training.pt``
        :param keep_percent: (float) Fraction of labels that are not set to be ignored
        :param train: (bool, optional) If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        :param transform: (callable, optional) A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        :param target_transform: (callable, optional) A function/transform that takes in the
            target and transforms it.
        :param download: (bool, optional) If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        :param ign: (int) Label which indicates non labeled examples
        :param showAll: (bool) Show sparse samples or only labeled ones
        """
        super(SparseMNIST, self).__init__(root, train, transform, target_transform, download)

        self.showAll = showAll
        N_samples = super(SparseMNIST, self).__len__()
        if self.showAll:
            self.n_sample = N_samples
        else:
            self.n_sample = int(keep_percent * N_samples)

        self.keep_percent = keep_percent
        self.ign = ign

        self.choice = torch.from_numpy(np.random.choice(
            range(N_samples),
            int(N_samples * self.keep_percent),
            replace=False))
        self.filter = torch.zeros(N_samples)
        self.filter[self.choice] = 1

    def __getitem__(self, index):
        if self.showAll:
            img, target = super(SparseMNIST, self).__getitem__(index)
            if not self.filter[index]:
                target = self.ign
        else:
            img, target = super(SparseMNIST, self).__getitem__(self.choice[index])

        return img, target

    def __len__(self):
        return self.n_sample
