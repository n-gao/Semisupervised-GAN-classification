import torch
import torch.nn as nn
import torch.nn.init as init


class MinibatchLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        """
        Implements Minibatch discrimination Layer from https://arxiv.org/pdf/1606.03498.pdf
        Note: Keep as small as possible as implemented with tensor of size in_features x out_features x kernel_dims

        :param in_features: Number of input feature maps
        :param out_features: Number of additional output feature maps
        :param kernel_dims:
        :param mean: Hidden Dimension
        """

        super().__init__()
        self.in_features = in_features  # A
        self.out_features = out_features  # B
        self.kernel_dims = kernel_dims  # C
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        """
        Adds statistics over all examples in minibatch to each sample
        Returns tensor of shape x.shape augmented by out_features in the channels direction

        :param x: Input
        :return: x after mbd
        """
        old_shape = x.size()
        x = x.view(-1, self.in_features)
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1
        o_b = o_b.view(old_shape[0], -1, old_shape[2], old_shape[3])
        x = x.view(old_shape[0], -1, old_shape[2], old_shape[3])

        x = torch.cat([x, o_b], 1)
        return x
