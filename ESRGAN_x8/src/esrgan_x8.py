"""Reference:
            ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (2018)
            Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang
"""

import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
        res_scale (float): Residual scale. Default: 0.2.
    """

    def __init__(self, features=64, res_scale=0.2, pytorch_init=True):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block

    Parameters:
        features (int): Channel number of intermediate VGGFeaturesExtractor.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, features=64, num_grow_ch=32, weights_init=True):
        super().__init__()
        self.conv1 = nn.Conv2d(features, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * num_grow_ch, features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if weights_init:
            init_weights(
                [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], scale=0.1
            )

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
        Real-ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
        We extend ESRGAN for scale x2, x4 and scale x8, where number of upscale
        n = 1, 2 and 3 represent the corresponding scale factor of x2, x4, and x8.

        Parameters:
            in_ch (int): Channel number of inputs. Default: 3
            out_ch (int): Channel number of out. Default: 3
            features (int): Channel number of intermediate VGGFeaturesExtractor. Default: 64
            upsample_factor (int): Scale factor. Default: 8 (x8)
            num_block (int): Block number in the body network. Defaults: 23
            num_grow_ch (int): Channels for each growth. Default: 32.
        """

    def __init__(
            self,
            in_ch=1,
            out_ch=1,
            features=64,
            upsample_factor=8,
            num_block=23,
            num_grow_ch=32
    ):
        super().__init__()
        RRDB_block = functools.partial(
            ResidualResidualDenseBlock, features=features, num_grow_ch=num_grow_ch
        )
        self.conv_first = nn.Conv2d(in_ch, features, 3, 1, 1)
        self.body = make_layer(RRDB_block, num_block)
        self.body_conv = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, out_ch, 3, 1, 1)

        # upsampling
        upsample_layers = []
        num_upscale = int(math.log(upsample_factor, 2))
        for _ in range(num_upscale):
            upsample_layers += [
                nn.Conv2d(features, features * 4, 3, 1, 1),
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


class Discriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    Parameters:
        in_ch (int): number of input channel. Default: 1.
        VGGFeaturesExtractor (int): number of base intermediate VGGFeaturesExtractor.Default: 64.
    """

    def __init__(self, input_shape=(1, 128, 128), upsample_factor=8):
        super().__init__()
        in_channels, in_height, in_width = input_shape
        n = int(math.log(upsample_factor, 2))
        patch_h, patch_w = int(in_height / 4 ** (n - 1)), int(in_width / 4 ** (n - 1))
        self.output_shape = (1, patch_h, patch_w)
        self.input_size = in_height

        blocks = []
        in_ch = in_channels
        for i, num_feature in enumerate([64, 128, 256, 512]):
            blocks.extend(self.discriminator_block(in_ch, num_feature, first_block=(i == 0)))
            in_ch = num_feature
        blocks.append(
            nn.Conv2d(num_feature, 1, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def discriminator_block(self, in_filters, out_filters, first_block):
        """Backbone of the discriminator blocks"""
        layers = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=True)
        ]

        if not first_block:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.extend(
            [nn.LeakyReLU(negative_slope=0.2, inplace=True),
             nn.Conv2d(out_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(out_filters, affine=True),
             nn.LeakyReLU(negative_slope=0.2, inplace=True)
             ]
        )

        return layers

    def forward(self, x):
        assert x.size(2) == self.input_size, \
            f"{Discriminator.__qualname__} takes input size {self.input_size}, but got {x.size(2)}."

        return self.blocks(x)


class VGGLoss(nn.Module):
    """Constructs a content loss function based on VGG19 network using
    VGGFeaturesExtractor before activation. Features are obtained by the 4th convolution
    before the 5th maxpooling layer
    """

    def __init__(self):
        super().__init__()
        vgg_19 = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg_19.features.children())[:35])
        self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg_loss = nn.L1Loss()

        for params in self.features.parameters():
            params.requires_grad = False

    def forward(self, sr, hr):
        sr = self.features(sr)
        hr = self.features(hr)
        return self.vgg_loss(sr, hr)


class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self._weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self._weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    netG = Generator(upsample_factor=8)
    x = torch.rand((1, 1, 16, 16))
    print(f"Generator output size: {netG(x).size()}")

    x = torch.rand((1, 1, 128, 128))
    netD = Discriminator((1, 128, 128), upsample_factor=8)
    print(f"Discriminator output size: {netD(x).size()}")

