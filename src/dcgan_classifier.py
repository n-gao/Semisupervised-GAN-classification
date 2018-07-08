import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dcgan_discriminator_multipleOut import Discriminator_MO
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from dcgan_discriminator_multipleOut import Discriminator_MO


class Classifier(nn.Module):
    def __init__(self, featureExtr, d=128):
        super(Classifier, self).__init__()
        self.featureExtr = featureExtr
        self.dropf = nn.Dropout2d(p=0.2)
        self.conv4_c = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn_c = nn.BatchNorm2d(d * 8)
        self.drop4 = nn.Dropout2d(p=0.4)
        self.conv5_c = nn.Conv2d(d * 8, 10, 4, 1, 0)

    def weight_init(self, mean, std):  # Will only iterate over new modules, not the feature extractor
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        feature = self.dropf(self.featureExtr.forward_3(input))
        c = F.leaky_relu(self.conv4_bn_c(self.conv4_c(feature)), 0.2)
        c = self.drop4(c)
        c = F.softmax(self.conv5_c(c))

        return c


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
