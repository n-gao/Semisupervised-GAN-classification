from torchvision import datasets

try:
    from .loader import loader
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from loader import loader

loader('cifar', datasets.CIFAR10, 3)
