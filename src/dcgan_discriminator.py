# based on
# https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/blob/master/pytorch_MNIST_cDCGAN.py

from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .minibatch_layer import MinibatchLayer
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from minibatch_layer import MinibatchLayer


class Discriminator(nn.Module):
    # initializers
    def __init__(self, channels=1, classes=10, conv_resolution=128, mbl_size=8, train_classes=True):
        super(Discriminator, self).__init__()

        if train_classes:
            self.label_conv = nn.Conv2d(classes, conv_resolution, kernel_size=4, stride=2, padding=1)
            self.conv1 = nn.Conv2d(channels, conv_resolution * 2, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(conv_resolution * 3, conv_resolution * 3, kernel_size=4, stride=2, padding=1)
            self.conv2_bn = nn.BatchNorm2d(conv_resolution * 3)
            self.conv3 = nn.Conv2d(conv_resolution * 3, conv_resolution * 4, kernel_size=4, stride=2, padding=1)
            self.conv3_bn = nn.BatchNorm2d(conv_resolution * 4)
        else:
            self.label_conv = None
            self.conv1 = nn.Conv2d(channels, conv_resolution * 2, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(conv_resolution * 2, conv_resolution * 2, kernel_size=4, stride=2, padding=1)
            self.conv2_bn = nn.BatchNorm2d(conv_resolution * 2)
            self.conv3 = nn.Conv2d(conv_resolution * 2, conv_resolution * 4, kernel_size=4, stride=2, padding=1)
            self.conv3_bn = nn.BatchNorm2d(conv_resolution * 4)
        self.conv4 = nn.Conv2d(conv_resolution * 4, conv_resolution * 8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(conv_resolution * 8)
        if mbl_size > 0:
            self.mbl = MinibatchLayer(conv_resolution * 8, mbl_size, 5)
        else:
            self.mbl = None
        self.conv5 = nn.Conv2d(conv_resolution * 8 + mbl_size, 1, kernel_size=4, stride=1, padding=0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1(input), 0.2)
        if self.label_conv is not None:
            y = F.leaky_relu(self.label_conv(label), 0.2)
            x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        feature = x
        if self.mbl is not None:
            x = self.mbl(x)
        x = F.sigmoid(self.conv5(x))

        return (x, feature)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
