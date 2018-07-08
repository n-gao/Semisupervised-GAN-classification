import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dcgan_discriminator import Discriminator
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from dcgan_discriminator import Discriminator


class Discriminator_MO(Discriminator):
    # initializers
    def __init__(self, channels=1, classes=10, conv_resolution=128, mbl_size=0, train_classes=True, freeze=True):
        super(Discriminator_MO, self).__init__(channels, classes, conv_resolution, mbl_size, train_classes)

        if self.label_conv is not None:
            print(self.label_conv)
            print("Pretraining with labels is cheating in semisupervised learning")
            raise NotImplementedError

        if freeze:
            for m in self.modules():
                m.requires_grad = False

    def forward_1(self, input):
        return F.leaky_relu(self.conv1(input), 0.2)

    def forward_2(self, input):
        x = self.forward_1(input)
        return F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)

    def forward_3(self, input):
        x = self.forward_2(input)
        return F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)

    def forward_4(self, input):
        x = self.forward_3(input)
        return F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
