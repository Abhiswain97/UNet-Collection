import torch
import torch.nn as nn
from torchvision.transforms import RandomCrop
from GenericBlocks import DoubleConvSame, Encoder, DoubleConv


class UNet(nn.Module):
    """Original UNet from the paper"""

    def __init__(self, c_in, c_out):
        super(UNet, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = Encoder(64)
        self.enc2 = Encoder(128)
        self.enc3 = Encoder(256)
        self.enc4 = Encoder(512)

        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConv(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConv(c_in=512, c_out=256)
        self.up_conv3 = DoubleConv(c_in=256, c_out=128)
        self.up_conv4 = DoubleConv(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

    def crop_tensor(self, up_tensor, target_tensor):
        _, _, H, W = up_tensor.shape

        x = RandomCrop(size=(H, W))(target_tensor)

        return x

    def forward(self, x):

        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER"""

        u1 = self.up1(c5)
        crop1 = self.crop_tensor(u1, c4)
        cat1 = torch.cat([u1, crop1], dim=1)
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        crop2 = self.crop_tensor(u2, c3)
        cat2 = torch.cat([u2, crop2], dim=1)
        uc2 = self.up_conv2(cat2)

        u3 = self.up3(uc2)
        crop3 = self.crop_tensor(u3, c2)
        cat3 = torch.cat([u3, crop3], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        crop4 = self.crop_tensor(u4, c1)
        cat4 = torch.cat([u4, crop4], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs

class UNet_2(nn.Module):
    """UNet with same padding"""
    
    def __init__(self, c_in, c_out):
        super(UNet_2, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = Encoder(64)
        self.enc2 = Encoder(128)
        self.enc3 = Encoder(256)
        self.enc4 = Encoder(512)

        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConvSame(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConvSame(c_in=512, c_out=256)
        self.up_conv3 = DoubleConvSame(c_in=256, c_out=128)
        self.up_conv4 = DoubleConvSame(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER"""

        u1 = self.up1(c5)
        cat1 = torch.cat([u1, c4], dim=1)
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        cat2 = torch.cat([u2, c3], dim=1)
        uc2 = self.up_conv2(cat2)

        u3 = self.up3(uc2)
        cat3 = torch.cat([u3, c2], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        cat4 = torch.cat([u4, c1], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs
