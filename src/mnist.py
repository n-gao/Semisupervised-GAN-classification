from torchvision import datasets

try:
    from .loader import loader
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from loader import loader

loader('mnist', datasets.MNIST, 1)
