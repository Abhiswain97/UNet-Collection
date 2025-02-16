import torch
from torchsummary import summary

from UNetOG import UNet_2
from AttentionUNet import AttentionUNet
from AttentionUNet3D import AttentionUNet3D

if __name__ == "__main__":

    unet = UNet_2(3, 1)
    attn_unet = AttentionUNet(3, 1)
    attn_unet_3d = AttentionUNet3D(3, 1)

    # print(summary(unet, input_data=(3, 256, 256))) 

    print(summary(attn_unet, (3, 32, 32)))

    # Uncomment this to print summary for attention unet
    # print(summary(attn_unet, input_data=(3, 256, 256)))
