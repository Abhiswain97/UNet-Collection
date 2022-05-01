from torchsummary import summary
from UNetOG import UNet
from AttentionUNet import AttentionUNet

if __name__ == "__main__":

    unet = UNet(3, 1)
    attn_unet = AttentionUNet(3, 1)

    print(summary(unet, input_data=(3, 256, 256)))

    # Uncomment this to print summary for attention unet
    # print(summary(attn_unet, input_data=(3, 256, 256)))
