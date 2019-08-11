import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def sample_noise(self, _):
        pass
    def remove_noise(self):
        pass

    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.img_shape = img_shape

    def forward(self, x, c, l):
        x = x.view(len(c), *self.img_shape)
        c = torch.cat([c, l], 1)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, img_shape, n_features, n_classes, space_dim, n_strided=3):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        n_strided -= 1

        def discriminator_block(in_filters, out_filters, norm):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.01))
            return layers

        layers = discriminator_block(channels, 64, False)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2, True))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        # PATCH GAN
        self.adv = nn.Conv2d(curr_dim, 1, 3, bias=False)
        self.cls = nn.Conv2d(curr_dim, 1, 3, stride=2, bias=False)

        with torch.no_grad():
            feats = self.model(torch.zeros(1, *img_shape))
            adv_dim = np.multiply.reduce(self.adv(feats).shape)
            cls_dim = np.multiply.reduce(self.cls(feats).shape)

        self.adv_out = nn.Linear(adv_dim, n_features, bias=False)
        self.cls_out = nn.Linear(cls_dim, space_dim)

        self.labels = nn.Conv2d(curr_dim, n_classes, kernel_size, bias=False)

        self.img_shape = img_shape

    def forward(self, img):
        feature_repr = self.model(img.view(-1, *self.img_shape))

#        print("SHAPES ", img.shape, feature_repr.shape)

        out_adv = self.adv(feature_repr).view(len(img), -1)
#        out_adv = torch.tanh(out_adv)
#        out_adv = self.adv_out(out_adv)
        out_cls = self.cls(feature_repr).view(len(img), -1)
        out_cls = self.cls_out(torch.relu(out_cls))
        out_labels = self.labels(feature_repr).view(len(img), -1)

#        print("\n ----> OUT SHAPES:", out_adv.shape, out_cls.shape, out_labels.shape, "<<<<\n")

#        print(torch.softmax(out_labels, 1))
        return (
                out_adv,
                out_cls,
                #  torch.softmax(out_labels, 1)
                nn.Softmax(dim=1)(out_labels)
                )
