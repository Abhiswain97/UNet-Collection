from torchsummary import summary
from UNetOG import UNet

if __name__ == "__main__":

    unet = UNet(3, 1)

    print(summary(unet, input_data=(3, 256, 256)))
