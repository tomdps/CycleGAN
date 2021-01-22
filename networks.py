import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from utils import *

class CycleGAN(nn.Module):
    def __init__(self, conv_features=64, norm_type='instance', init='normal', n_blocks=9, input_channel=3, output_channel=3):
        super(CycleGAN, self).__init__()
        norm_layer = normalization_layer(norm_type)
        self.G1 = Generator(input_channel, output_channel, norm_layer, conv_features, n_blocks)
        self.G2 = Generator(input_channel, output_channel, norm_layer, conv_features, n_blocks)
        self.D1 = Discriminator(output_channel, norm_layer, conv_features)
        self.D2 = Discriminator(output_channel, norm_layer, conv_features)
        self.g_opt = None
        self.d_opt = None


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super(ResidualBlock, self).__init__()
        bias = norm_layer.func == nn.InstanceNorm2d
        resblock = [
            nn.ReflectionPad2d(1), 
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=bias), 
            norm_layer(dim), 
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=bias),
            norm_layer(dim)
            ]
        self.resblock = nn.Sequential(*resblock)
    
    def forward(self, x):
        out = x + self.resblock(x)
        return out

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, norm_layer, conv_features=64, n_blocks=9):
        super(Generator, self).__init__()
        bias = norm_layer.func == nn.InstanceNorm2d
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel, conv_features, kernel_size=7, padding=0, bias=bias),
            norm_layer(conv_features),
            nn.ReLU(True)
            ]

        model += [
            nn.Conv2d(conv_features, conv_features * 2, kernel_size=3, stride=2, padding=1, bias=bias),
            norm_layer(conv_features * 2),
            nn.ReLU(True),
            nn.Conv2d(conv_features * 2, conv_features * 4, kernel_size=3, stride=2, padding=1, bias=bias),
            norm_layer(conv_features * 4),
            nn.ReLU(True)
                    ]

        for i in range(n_blocks):
            model += [ResidualBlock(conv_features * 4, norm_layer)]

        model += [
            nn.ConvTranspose2d(conv_features * 4, conv_features * 2, kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=bias),
            norm_layer(int(conv_features * 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_features * 2, conv_features, kernel_size=3, stride=2,
                                padding=1, output_padding=1, bias=bias),
            norm_layer(conv_features),
            nn.ReLU(True)
                    ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(conv_features, output_channel, kernel_size=7, padding=0),
            nn.Tanh()
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channel, norm_layer, conv_features=64):
        super().__init__()
        bias = norm_layer.func == nn.InstanceNorm2d
        model = [nn.Conv2d(input_channel, conv_features, kernel_size=4, stride=2, padding=1), 
                nn.LeakyReLU(0.2, True)]

        model += [
            nn.Conv2d(conv_features, conv_features * 2, kernel_size=4, stride=2, padding=1, bias=bias),
            norm_layer(conv_features * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_features * 2, conv_features * 4, kernel_size=4, stride=2, padding=1, bias=bias),
            norm_layer(conv_features * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_features * 4, conv_features * 8, kernel_size=4, stride=2, padding=1, bias=bias),
            norm_layer(conv_features * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_features * 8, conv_features * 8, kernel_size=4, stride=1, padding=1, bias=bias),
            norm_layer(conv_features * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_features * 8, 1, kernel_size=4, stride=1, padding=1)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = CycleGAN()
    model.train()
