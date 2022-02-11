import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

class ISINET(nn.Module):
    def __init__(self, N, ampli=2, reduction=2, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        # Given that an Ising model only deals with short-range interactions,
        #    small-sized 2D convolutions should do the trick.

        # After convolution dimensions
        pad = 0
        L_out = lambda L_in, k, s, dil, pad=pad: int(((L_in+2*pad-dil*(k-1)-1)/s)+1) 
        first_conv = L_out(N, kernel_size, stride, dilation)
        second_conv = L_out(first_conv, kernel_size, stride, dilation)
        mlp_in = int((ampli**2)*(second_conv**2))
        first_red = mlp_in//reduction
        second_red = first_red//reduction

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=ampli, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation),
            nn.BatchNorm2d(ampli),
            nn.ReLU(),
            nn.Conv2d(in_channels=ampli, out_channels=ampli*2, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation),
            nn.BatchNorm2d(ampli*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=mlp_in, out_features=first_red),
            nn.ReLU(),
            nn.Linear(in_features=first_red, out_features=second_red),
            nn.ReLU(),
            nn.Linear(in_features=second_red, out_features=1) # The energy is a scalar
            )

    def forward(self, x):
        return self.mlp(self.conv(x))

class FREENET(nn.Module):
    def __init__(self) -> None:
        super().__init__()