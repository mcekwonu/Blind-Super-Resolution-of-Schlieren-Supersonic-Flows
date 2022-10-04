"""Ref: RealESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (2021)
        Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan
"""

import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm
from torchvision.models import vgg19


def make_layer(basic_block, num_block, **kwargs):
    """Make layers by stacking the same blocks.

    Parameters:
        basic_block (nn.module): class for basic block.
        num_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = [
        basic_block(**kwargs) for _ in range(num_block)
    ]

    return nn.Sequential(*layers)


@torch.no_grad()
def init_weights(module_list, scale=0.1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Parameters:
        module_list (list[nn.Module]): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight *= scale
                if m.bias is not None:
                    m.bias.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight *= scale
                if m.bias is not None:
                    m.bias.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Parameters:
        features (int): Channel number of intermediate VGGFeaturesExtractor.
            Default: 64.
        residual_scale (float): Residual scale. Default: 0.2.
    """

    def __init__(self, features=64, residual_scale=0.2, pytorch_init=True):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            init_weights([self.conv1, self.conv2], scale=0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.residual_scale


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block

    Parameters:
        features (int): Channel number of intermediate VGGFeaturesExtractor.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, features=64, num_grow_ch=32, weight_init=True):
        super().__init__()
        self.conv1 = nn.Conv2d(features, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * num_grow_ch, features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if weight_init:
            init_weights([self.conv1, self.conv2], scale=0.1)
            init_weights([self.conv3, self.conv4, self.conv5], scale=1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class ResidualResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block

    Parameters:
        features (int): Channel number of intermediate VGGFeaturesExtractor.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, features, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(features, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(features, num_grow_ch)

    def forward(self, x):
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + identity


class Generator(nn.Module):
    """Residual in Residual Dense Block Network.
        RealESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
        We extend ESRGAN for scale x2, x4 and scale x8, where number of upscale
        n = 1, 2 and 3 represent the corresponding scale factor of x2, x4, and x8.

        Parameters:
            in_ch (int): Channel number of inputs. Default: 3
            out_ch (int): Channel number of out. Default: 3
            features (int): Channel number of intermediate VGGFeaturesExtractor. Default: 64
            upsample_factor (int): Scale factor. Default: 4 (x4)
            num_block (int): Block number in the body network. Defaults: 23
            num_grow_ch (int): Channels for each growth. Default: 32.
        """

    def __init__(
            self,
            in_ch=1,
            out_ch=1,
            features=64,
            upsample_factor=4,
            num_block=23,
            num_grow_ch=32
    ):
        super().__init__()
        RRDB_block = functools.partial(
            ResidualResidualDenseBlock, features=features, num_grow_ch=num_grow_ch
        )
        self.conv_first = nn.Conv2d(in_ch, features, kernel_size=3, stride=1, padding=1)
        self.body = make_layer(RRDB_block, num_block)
        self.body_conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv_hr = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(features, out_ch, kernel_size=3, stride=1, padding=1)

        # upsampling
        upsample_layers = []
        n = int(math.log(upsample_factor, 2))
        for _ in range(n):
            upsample_layers += [
                nn.Conv2d(features, features * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.body_conv(self.body(feat))
        feat = feat + body_feat
        feat = self.upsampling(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out


class UNetDiscriminator(nn.Module):
    """UNet-style Discriminator with Spectral normalization (SN).

    Parameters:
        input_shape tuple(int): Input shape. Default is (1, 64, 64).
        num_feat (int): Number of VGGFeaturesExtractor. Default is 64.
        skip_connection (bool): Skip connections between encoder and decoder branch. Default is True.
    """

    def __init__(self, input_shape=(1, 64, 64), num_feat=64, skip_connection=True):
        super().__init__()
        self.input_shape = input_shape
        self.skip_connection = skip_connection
        norm = spectral_norm
        in_channels, _, _ = input_shape

        self.conv0 = nn.Conv2d(in_channels, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=4, stride=2, padding=1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv9 = norm(nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        assert x.size(2) == self.input_shape[2], \
            f"{UNetDiscriminator.__qualname__} takes input size of {self.input_shape[2]}, but got {x.size(2)}"

        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class PerceptualLoss(nn.Module):
    """Constructs a content loss function based on VGG19 network using
    VGGFeaturesExtractor before activation. Features are obtained by the 4th convolution
    before the 5th maxpooling layer
    """

    def __init__(self):
        super().__init__()
        vgg_19 = vgg19(pretrained=True)
        self.VGGFeaturesExtractor = nn.Sequential(*list(vgg_19.features.children())[:35])
        self.VGGFeaturesExtractor[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.VGGFeaturesExtractor[2].weight * 0.1
        self.VGGFeaturesExtractor[7].weight * 0.1
        self.criterion = nn.L1Loss()

        for params in self.VGGFeaturesExtractor.parameters():
            params.requires_grad = False

    def forward(self, sr, hr):
        sr = self.VGGFeaturesExtractor(sr)
        hr = self.VGGFeaturesExtractor(hr)

        return self.criterion(sr, hr)


if __name__ == "__main__":

    x = torch.rand((1, 1, 64, 64))
    x1 = torch.rand((1, 1, 16, 16))
    netD = UNetDiscriminator(input_shape=(1, 64, 64))
    print(f'Discriminator output size: {netD(x).size()}')

    netG = Generator(upsample_factor=4)
    print(f'Generator output size: {netG(x1).size()}')

    PerceptualLoss()

