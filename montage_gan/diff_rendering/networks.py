import math

import torch.nn as nn


class Renderer(nn.Module):
    def __init__(self,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 img_layers,  # Number of input image layers.
                 nf=64,
                 ):
        super(Renderer, self).__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.img_layers = img_layers

        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # A very simple CNN, each block's input output are the same size
        self.cnn = nn.Sequential(
            # Input layer
            nn.Conv2d(img_channels * img_layers, nf, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 2nd ~ 4th layers
            *[self.block] * 3,
            # Output layer
            nn.Conv2d(nf, img_channels, kernel_size=3, padding=1),
            # Sigmoid
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input: [B,L,C,H,W]
        Output: [B,C,H,W]
        """
        b, l, c, h, w = x.shape
        x = x.view(b, l * c, h, w)  # [B,L*C,H,W]
        out = self.cnn(x)  # [B,C,H,W]
        return out


class RendererTanh(nn.Module):
    """
    Another variant of renderer that output [-1,1]
    """

    def __init__(self,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 img_layers,  # Number of input image layers.
                 nf=64,
                 ):
        super(RendererTanh, self).__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.img_layers = img_layers

        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # A very simple CNN, each block's input output are the same size
        self.cnn = nn.Sequential(
            # Input layer
            nn.Conv2d(img_channels * img_layers, nf, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 2nd ~ 4th layers
            *[self.block] * 3,
            # Output layer
            nn.Conv2d(nf, img_channels, kernel_size=3, padding=1),
            # Tanh
            nn.Tanh()
        )

    def forward(self, x):
        """
        Input: [B,L,C,H,W]
        Output: [B,C,H,W]
        """
        b, l, c, h, w = x.shape
        x = x.view(b, l * c, h, w)  # [B,L*C,H,W]
        out = self.cnn(x)  # [B,C,H,W]
        return out


class RendererSubPixelConv(nn.Module):
    def __init__(self,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 img_layers,  # Number of input image layers.
                 nf1=8,
                 nf2=64,
                 ):
        super(RendererSubPixelConv, self).__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.img_layers = img_layers

        r = math.isqrt(img_channels * img_layers)
        assert r ** 2 == img_channels * img_layers
        assert r == 6  # Currently only support 9 layers of RGBA image

        self.block = nn.Sequential(
            nn.Conv2d(nf2, nf2, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.cnn = nn.Sequential(
            nn.PixelShuffle(r),  # [B,36,H,W] -> [B,1,6H,6W]
            nn.Conv2d(1, nf1, kernel_size=3, stride=2, padding=1),  # [B,1,6H,6W] -> [B,nf,3H,3W]
            nn.ReLU(True),
            nn.Conv2d(nf1, nf2, kernel_size=3, stride=3),  # [B,nf,3H,3W] -> [B,nf,H,W]
            nn.ReLU(True),
            *[self.block] * 2,  # [B,nf,H,W] -> [B,nf,H,W]
            nn.Conv2d(nf2, 4, kernel_size=3, padding=1),
            # Tanh
            nn.Tanh()
        )

    def forward(self, x):
        """
        Input: [B,L,C,H,W]
        Output: [B,C,H,W]
        """
        b, l, c, h, w = x.shape
        x = x.view(b, l * c, h, w)  # [B,L*C,H,W]
        out = self.cnn(x)  # [B,C,H,W]
        return out


if __name__ == "__main__":
    # from torchinfo import summary

    # Test the network with dummy data
    # renderer = Renderer(256, 4, 9)

    # Test RendererSubPixelConv with dummy data
    # renderer = RendererSubPixelConv(256, 4, 9)
    renderer = Renderer(256, 4, 9)
    # summary(renderer, input_size=(8, 9, 4, 256, 256))

    print(renderer)
    # dummy = torch.randn(size=(8, 9, 4, 256, 256))
    # out = renderer(dummy)
    # print(out, out.shape)
