# based on
# https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/blob/master/pytorch_MNIST_cDCGAN.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, channels=1, conv_resolution=128, train_classes=True):
        super(Generator, self).__init__()
        if train_classes:
            self.deconv1 = nn.ConvTranspose2d(100, conv_resolution * 4, kernel_size=4, stride=1, padding=0)
            self.deconv1_bn = nn.BatchNorm2d(conv_resolution * 4)
            self.label_deconv = nn.ConvTranspose2d(10, conv_resolution * 4, kernel_size=4, stride=1, padding=0)
            self.label_deconv_bn = nn.BatchNorm2d(conv_resolution * 4)
        else:
            self.label_deconv = None
            self.deconv1 = nn.ConvTranspose2d(100, conv_resolution * 8, kernel_size=4, stride=1, padding=0)
            self.deconv1_bn = nn.BatchNorm2d(conv_resolution * 8)
        self.deconv2 = nn.ConvTranspose2d(conv_resolution * 8, conv_resolution * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(conv_resolution * 4)
        self.deconv3 = nn.ConvTranspose2d(conv_resolution * 4, conv_resolution * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3_bn = nn.BatchNorm2d(conv_resolution * 2)
        self.deconv4 = nn.ConvTranspose2d(conv_resolution * 2, conv_resolution, kernel_size=4, stride=2, padding=1)
        self.deconv4_bn = nn.BatchNorm2d(conv_resolution)
        self.deconv5 = nn.ConvTranspose2d(conv_resolution, channels, kernel_size=4, stride=2, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        if self.label_deconv is not None:
            y = F.relu(self.label_deconv_bn(self.label_deconv(label)))
            x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
