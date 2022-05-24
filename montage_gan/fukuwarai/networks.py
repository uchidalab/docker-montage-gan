import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_utils.image_utils import convert_translate_to_2x3


# class STN(nn.Module):
#     def __init__(self,
#                  img_resolution,  # Input resolution.
#                  img_channels,  # Number of input color channels.
#                  img_layers,  # Number of input image layers.
#                  nf1=64,
#                  nf2=64,
#                  ):
#         super(STN, self).__init__()
#         self.img_resolution = img_resolution
#         self.img_channels = img_channels
#         self.img_layers = img_layers
#
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(img_channels * img_layers, nf1, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(nf1, nf1 * 2, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         self.len_loc = len(
#             self.localization(torch.randn(1, img_channels * img_layers, img_resolution, img_resolution)).flatten())
#
#         # Regression for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(self.len_loc, nf2),
#             nn.ReLU(True),
#             nn.Linear(nf2, img_layers * 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0] * img_layers, dtype=torch.float))
#
#     def forward(self, x):
#         """
#         Input: [B,L,C,H,W] (resolution 256*256, 9 layers, RGBA expected)
#         Output: [B,L,C,H,W] (transformed), [B,L,2,3] (transformation parameters)
#         """
#
#         b, l, c, h, w = x.shape
#         x1 = x.view(b, l * c, h, w)  # [B,L*C,H,W]
#         # Predict the transformation parameters
#         x1 = self.localization(x1)
#         x1 = x1.view(-1, self.len_loc)
#         theta = self.fc_loc(x1)  # [B,L*2*3]
#         theta = theta.view(-1, 2, 3)  # [B*L,2,3]
#
#         x2 = x.view(-1, c, h, w)  # [B*L,C,H,W]
#         grid = F.affine_grid(theta, x2.size(), align_corners=False)
#         x2 = F.grid_sample(x2, grid, align_corners=False)
#
#         # Reformat output
#         x, theta = x2.view(b, l, c, h, w), theta.view(b, l, 2, 3)
#         return x, theta


# class STNv2(nn.Module):
#     """
#     v2: Further constrained the network to output only the translation
#     Also adding tanh to the end of the network, so no other constraint is needed
#     Also adding more hidden layer to the localization
#     """
#
#     def __init__(self,
#                  img_resolution,  # Input resolution.
#                  img_channels,  # Number of input color channels.
#                  img_layers,  # Number of input image layers.
#                  nf1=64,
#                  nf2=64,
#                  ):
#         super(STNv2, self).__init__()
#         self.img_resolution = img_resolution
#         self.img_channels = img_channels
#         self.img_layers = img_layers
#
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(img_channels * img_layers, nf1, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             #
#             nn.Conv2d(nf1, nf1 * 2, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             #
#             nn.Conv2d(nf1 * 2, nf1 * 4, kernel_size=3),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             #
#             nn.Conv2d(nf1 * 4, nf1 * 6, kernel_size=3),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             #
#             nn.Conv2d(nf1 * 6, nf1 * 8, kernel_size=3),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#         )
#
#         self.len_loc = len(
#             self.localization(torch.randn(1, img_channels * img_layers, img_resolution, img_resolution)).flatten())
#
#         # Regression for the translation
#         self.fc_loc = nn.Sequential(
#             nn.Linear(self.len_loc, nf2),
#             nn.ReLU(True),
#             nn.Linear(nf2, img_layers * 2),
#             nn.Tanh()
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.zero_()
#
#     def forward(self, x):
#         """
#         Input: [B,L,C,H,W] (resolution 256*256, 9 layers, RGBA expected)
#         Output: [B,L,C,H,W] (transformed), [B,L,2,3] (transformation parameters)
#         """
#
#         b, l, c, h, w = x.shape
#         x1 = x.view(b, l * c, h, w)  # [B,L*C,H,W]
#         # Predict the transformation parameters
#         x1 = self.localization(x1)
#         x1 = x1.view(-1, self.len_loc)
#         translation = self.fc_loc(x1).view(b, l, 2)
#         theta = convert_translate_to_2x3(translation)  # [B,L,2,3]
#         theta = theta.view(-1, 2, 3)  # [B*L,2,3]
#
#         x2 = x.view(-1, c, h, w)  # [B*L,C,H,W]
#         grid = F.affine_grid(theta, x2.size(), align_corners=False)
#         x2 = F.grid_sample(x2, grid, align_corners=False)
#
#         # Reformat output
#         x, theta = x2.view(b, l, c, h, w), theta.view(b, l, 2, 3)
#         return x, theta

class STNv2b(nn.Module):
    """
    v2b: Further constrained the network to output only the translation
    Also adding more hidden layer to the localization
    Removed tanh from the end of the network, as it seems to cause layers to be stuck at the corner
    With this version, theta constrain is needed
    """

    def __init__(self,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 img_layers,  # Number of input image layers.
                 nf1=64,
                 nf2=64,
                 ):
        super(STNv2b, self).__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.img_layers = img_layers

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(img_channels * img_layers, nf1, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            #
            nn.Conv2d(nf1, nf1 * 2, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            #
            nn.Conv2d(nf1 * 2, nf1 * 4, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            #
            nn.Conv2d(nf1 * 4, nf1 * 6, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            #
            nn.Conv2d(nf1 * 6, nf1 * 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.len_loc = len(
            self.localization(torch.randn(1, img_channels * img_layers, img_resolution, img_resolution)).flatten())

        # Regression for the translation
        self.fc_loc = nn.Sequential(
            nn.Linear(self.len_loc, nf2),
            nn.ReLU(True),
            nn.Linear(nf2, img_layers * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

    def forward(self, x):
        """
        Input: [B,L,C,H,W] (resolution 256*256, 9 layers, RGBA expected)
        Output: [B,L,C,H,W] (transformed), [B,L,2,3] (transformation parameters)
        """

        b, l, c, h, w = x.shape
        x1 = x.view(b, l * c, h, w)  # [B,L*C,H,W]
        # Predict the transformation parameters
        x1 = self.localization(x1)
        x1 = x1.view(-1, self.len_loc)
        translation = self.fc_loc(x1).view(b, l, 2)
        theta = convert_translate_to_2x3(translation)  # [B,L,2,3]
        theta = theta.view(-1, 2, 3)  # [B*L,2,3]

        x2 = x.view(-1, c, h, w)  # [B*L,C,H,W]
        grid = F.affine_grid(theta, x2.size(), align_corners=False)
        x2 = F.grid_sample(x2, grid, align_corners=False)

        # Reformat output
        x, theta = x2.view(b, l, c, h, w), theta.view(b, l, 2, 3)
        return x, theta


class STNv2c(STNv2b):
    """
    v2c: Another variant for training data that ranged in [-1,1]
    Using workaround https://issueexplorer.com/issue/pytorch/pytorch/66366 for grid_sample
    """

    def forward(self, x):
        """
        Input: [B,L,C,H,W] (resolution 256*256, 9 layers, RGBA expected)
        Output: [B,L,C,H,W] (transformed), [B,L,2,3] (transformation parameters)
        """

        b, l, c, h, w = x.shape
        x1 = x.view(b, l * c, h, w)  # [B,L*C,H,W]
        # Predict the transformation parameters
        x1 = self.localization(x1)
        x1 = x1.view(-1, self.len_loc)
        translation = self.fc_loc(x1).view(b, l, 2)
        theta = convert_translate_to_2x3(translation)  # [B,L,2,3]
        theta = theta.view(-1, 2, 3)  # [B*L,2,3]

        x2 = x.view(-1, c, h, w)  # [B*L,C,H,W]
        grid = F.affine_grid(theta, x2.size(), align_corners=False)
        # Workaround
        x2 = x2 + 1  # [0,2]
        x2 = F.grid_sample(x2, grid, align_corners=False)
        x2 = x2 - 1  # [-1,1]
        # Reformat output
        x, theta = x2.view(b, l, c, h, w), theta.view(b, l, 2, 3)
        return x, theta


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        p = kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size, padding=p),
            nn.LeakyReLU(0.2)
        )
        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.conv2(y)

        identity = identity if self.proj is None else self.proj(identity)
        y = y + identity
        return y


class SimpleGlobalDiscriminator(nn.Module):
    """
    A very simple Global Discriminator
    Ref: https://github.com/Yangyangii/GAN-Tutorial/blob/master/CelebA/R1GAN.ipynb
    Ref: https://arxiv.org/pdf/1801.04406.pdf, Table 3
    """

    def __init__(self,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 nf1=64,
                 ):
        super(SimpleGlobalDiscriminator, self).__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.D = nn.Sequential(
            nn.Conv2d(img_channels, nf1, 3, padding=1),  # [64x256x256]
            ResidualBlock(nf1, nf1),
            nn.AvgPool2d(3, 2, padding=1),  # [64x128x128]
            ResidualBlock(nf1, nf1 * 2),
            nn.AvgPool2d(3, 2, padding=1),  # [128x64x64]
            ResidualBlock(nf1 * 2, nf1 * 4),
            nn.AvgPool2d(3, 2, padding=1),  # [256x32x32]
            ResidualBlock(nf1 * 4, nf1 * 8),
            nn.AvgPool2d(3, 2, padding=1),  # [512x16x16]
            ResidualBlock(nf1 * 8, nf1 * 16),
            nn.AvgPool2d(3, 2, padding=1),  # [1024x8x8]
            ResidualBlock(nf1 * 16, nf1 * 16),
            nn.AvgPool2d(3, 2, padding=1),  # [1024x4x4]
        )
        self.len_cnn = len(
            self.D(torch.randn(1, img_channels, img_resolution, img_resolution)).flatten())
        self.fc = nn.Linear(self.len_cnn, 1)  # [1]

    def forward(self, x):
        B = x.size(0)
        h = self.D(x)
        h = h.view(B, -1)
        y = self.fc(h)
        return y


class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.mode = "bilinear"

    def forward(self, x):
        h, w = x.shape[-2:]
        x = nn.functional.interpolate(x, size=(h // 2, w // 2), mode=self.mode, align_corners=False)
        return x


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, inplanes, tmp_planes, planes, img_channels, kernel_size=3, stride=1):
        super(DiscriminatorBlock, self).__init__()
        self.img_channels = img_channels
        self.inplanes = inplanes
        self.fromrgb = nn.Sequential()

        p = kernel_size // 2
        if inplanes == 0:
            self.fromrgb = nn.Sequential(
                nn.Conv2d(self.img_channels, tmp_planes, kernel_size=1, stride=stride, padding=0),
                nn.LeakyReLU(0.2),
            )
        self.conv0 = nn.Sequential(
            nn.Conv2d(tmp_planes, tmp_planes, kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(tmp_planes, planes, kernel_size, stride=stride, padding=p),
            # nn.AvgPool2d(3, 2, padding=1),  # Down-sampling
            DownSampling(),
            nn.LeakyReLU(0.2)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(tmp_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
            # nn.AvgPool2d(3, 2, padding=1),  # Down-sampling
            DownSampling()
        )

    def forward(self, x):
        # FromRGB.
        if self.inplanes == 0:
            x = self.fromrgb(x)

        # Main layers.
        y = self.skip(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = y.add_(x)

        return x


class SimpleGlobalDiscriminatorV2(nn.Module):
    """
    Global Discriminator v2, closer implementation with SG2ada's
    """

    def __init__(self,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 nf=512,
                 ):
        super(SimpleGlobalDiscriminatorV2, self).__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.D = nn.Sequential(
            DiscriminatorBlock(0, 128, 256, img_channels),
            DiscriminatorBlock(256, 256, 512, img_channels),
            DiscriminatorBlock(512, 512, 512, img_channels),
            DiscriminatorBlock(512, 512, 512, img_channels),
            DiscriminatorBlock(512, 512, 512, img_channels),
            DiscriminatorBlock(512, 512, 512, img_channels),
        )
        self.len_cnn = len(
            self.D(torch.randn(1, img_channels, img_resolution, img_resolution)).flatten())
        self.fc = nn.Sequential(
            nn.Linear(self.len_cnn, nf),
            nn.Linear(nf, 1)
        )  # [1]

    def forward(self, x):
        B = x.size(0)
        h = self.D(x)
        h = h.view(B, -1)
        y = self.fc(h)
        return y


def test_stn():
    stn = STNv2b(256, 4, 9).to("cuda")
    print(stn)
    dummy = torch.randn(size=(8, 9, 4, 256, 256), device="cuda")
    out, t = stn(dummy)
    print(out, t, out.shape)


def test_discriminator():
    global_d = SimpleGlobalDiscriminator(256, 4).to("cuda")
    print(global_d)
    dummy = torch.randn(size=(8, 4, 256, 256), device="cuda")
    out = global_d(dummy)
    print(out)


if __name__ == "__main__":
    # Test the network with dummy data
    test_stn()
    test_discriminator()
    # from torchinfo import summary
    #
    # d = SimpleGlobalDiscriminator(256, 4)
    # print(summary(d, input_size=(16, 4, 256, 256)))

    # from torchinfo import summary
    #
    # d = SimpleGlobalDiscriminatorV2(256, 4)
    # print(summary(d, input_size=(16, 4, 256, 256), depth=4))
