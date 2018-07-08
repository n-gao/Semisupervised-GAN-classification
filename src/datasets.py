from torchvision import datasets
import torch
import numpy as np

class SparseMNIST(datasets.MNIST):
    """Replaces part of the original MNIST labels with an ignore label
    """
    def __init__(self, root, keep_percent, train=True, transform=None, target_transform=None, download=False, ign = -100):
        super(SparseMNIST, self).__init__(root, train, transform, target_transform, download)
        
        self.n_sample = super(SparseMNIST, self).__len__()
        self.keep_percent = keep_percent
        self.ign = ign
        
        choice = torch.from_numpy(np.random.choice(range(self.n_sample), int(self.n_sample*self.keep_percent), replace=False))
        self.filter = torch.zeros(self.n_sample)
        self.filter[choice] = 1

    def __getitem__(self, index):
        img, target = super(SparseMNIST, self).__getitem__(index)
        if not self.filter[index]:
            target = self.ign
            
        return img, target
        
class LabeledMNIST(datasets.MNIST):
    """Replaces part of the original MNIST labels with an ignore label
    """
    def __init__(self, root, keep_percent, train=True, transform=None, target_transform=None, download=False):
        super(LabeledMNIST, self).__init__(root, train, transform, target_transform, download)
        
        self.n_sample = super(LabeledMNIST, self).__len__()
        self.keep_percent = keep_percent

        self.len = int(self.n_sample * self.keep_percent)

    def __getitem__(self, index):
        img, target = super(LabeledMNIST, self).__getitem__(index)      
        return img, target

    def __len__(self):
        return self.len

class UnlabeledMNIST(datasets.MNIST):
    """Replaces part of the original MNIST labels with an ignore label
    """
    def __init__(self, root, keep_percent, train=True, transform=None, target_transform=None, download=False):
        super(UnlabeledMNIST, self).__init__(root, train, transform, target_transform, download)
        
        self.n_sample = super(UnlabeledMNIST, self).__len__()
        self.keep_percent = keep_percent

        self.off = int(self.n_sample * (1 - self.keep_percent))
        self.len = int(self.n_sample * self.keep_percent)

    def __getitem__(self, index):
        img, target = super(UnlabeledMNIST, self).__getitem__(self.off + index)      
        return img, target

    def __len__(self):
        return self.len
