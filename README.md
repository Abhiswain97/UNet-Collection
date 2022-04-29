# UNet-Collection
Contains implementations of Original UNet, Attention-UNet, 3D-UNet and Attention-UNet-3D in PyTorch

| Paper | Code |
| ----- | ---- |
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | [UNetOG.py](https://github.com/Abhiswain97/UNet-Collection/blob/main/UNetOG.py) |
| [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650.pdf) | [UNet3D.py](https://github.com/Abhiswain97/UNet-Collection/blob/main/UNet3D.py) |
| [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999.pdf) | [AttentionUnet.py](https://github.com/Abhiswain97/UNet-Collection/blob/main/AttentionUNet.py) |
| [Brain Tumor Segmentation and Survival Prediction using 3D Attention UNet](https://arxiv.org/pdf/2104.00985.pdf) | [AttentionUNet3D.py](https://github.com/Abhiswain97/UNet-Collection/blob/main/AttentionUNet3D.py) |

## Viewing the summary of a model

Install the requirements using: `pip install requirements.txt`

Run the file `main.py` using: `python main.py`
By default you should see the summary of the Original UNet

You can play around with it by importing the other UNets and seeing their summaries.

Summary of UNet: 
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─DoubleConvSame: 1-1                    [-1, 64, 256, 256]        --
|    └─Sequential: 2-1                   [-1, 64, 256, 256]        --
|    |    └─Conv2d: 3-1                  [-1, 64, 256, 256]        1,792
|    |    └─ReLU: 3-2                    [-1, 64, 256, 256]        --
|    |    └─Conv2d: 3-3                  [-1, 64, 256, 256]        36,928
|    |    └─ReLU: 3-4                    [-1, 64, 256, 256]        --
├─MaxPool2d: 1-2                         [-1, 64, 128, 128]        --
├─Encoder: 1-3                           [-1, 128, 128, 128]       --
|    └─DoubleConvSame: 2-2               [-1, 128, 128, 128]       --
|    |    └─Sequential: 3-5              [-1, 128, 128, 128]       221,440
|    └─MaxPool2d: 2-3                    [-1, 128, 64, 64]         --
├─Encoder: 1-4                           [-1, 256, 64, 64]         --
|    └─DoubleConvSame: 2-4               [-1, 256, 64, 64]         --
|    |    └─Sequential: 3-6              [-1, 256, 64, 64]         885,248
|    └─MaxPool2d: 2-5                    [-1, 256, 32, 32]         --
├─Encoder: 1-5                           [-1, 512, 32, 32]         --
|    └─DoubleConvSame: 2-6               [-1, 512, 32, 32]         --
|    |    └─Sequential: 3-7              [-1, 512, 32, 32]         3,539,968
|    └─MaxPool2d: 2-7                    [-1, 512, 16, 16]         --
├─DoubleConvSame: 1-6                    [-1, 1024, 16, 16]        --
|    └─Sequential: 2-8                   [-1, 1024, 16, 16]        --
|    |    └─Conv2d: 3-8                  [-1, 1024, 16, 16]        4,719,616
|    |    └─ReLU: 3-9                    [-1, 1024, 16, 16]        --
|    |    └─Conv2d: 3-10                 [-1, 1024, 16, 16]        9,438,208
|    |    └─ReLU: 3-11                   [-1, 1024, 16, 16]        --
├─ConvTranspose2d: 1-7                   [-1, 512, 32, 32]         2,097,664
├─DoubleConv: 1-8                        [-1, 512, 28, 28]         --
|    └─Sequential: 2-9                   [-1, 512, 28, 28]         --
|    |    └─Conv2d: 3-12                 [-1, 512, 30, 30]         4,719,104
|    |    └─ReLU: 3-13                   [-1, 512, 30, 30]         --
|    |    └─Conv2d: 3-14                 [-1, 512, 28, 28]         2,359,808
|    |    └─ReLU: 3-15                   [-1, 512, 28, 28]         --
├─ConvTranspose2d: 1-9                   [-1, 256, 56, 56]         524,544
├─DoubleConv: 1-10                       [-1, 256, 52, 52]         --
|    └─Sequential: 2-10                  [-1, 256, 52, 52]         --
|    |    └─Conv2d: 3-16                 [-1, 256, 54, 54]         1,179,904
|    |    └─ReLU: 3-17                   [-1, 256, 54, 54]         --
|    |    └─Conv2d: 3-18                 [-1, 256, 52, 52]         590,080
|    |    └─ReLU: 3-19                   [-1, 256, 52, 52]         --
├─ConvTranspose2d: 1-11                  [-1, 128, 104, 104]       131,200
├─DoubleConv: 1-12                       [-1, 128, 100, 100]       --
|    └─Sequential: 2-11                  [-1, 128, 100, 100]       --
|    |    └─Conv2d: 3-20                 [-1, 128, 102, 102]       295,040
|    |    └─ReLU: 3-21                   [-1, 128, 102, 102]       --
|    |    └─Conv2d: 3-22                 [-1, 128, 100, 100]       147,584
|    |    └─ReLU: 3-23                   [-1, 128, 100, 100]       --
├─ConvTranspose2d: 1-13                  [-1, 64, 200, 200]        32,832
├─DoubleConv: 1-14                       [-1, 64, 196, 196]        --
|    └─Sequential: 2-12                  [-1, 64, 196, 196]        --
|    |    └─Conv2d: 3-24                 [-1, 64, 198, 198]        73,792
|    |    └─ReLU: 3-25                   [-1, 64, 198, 198]        --
|    |    └─Conv2d: 3-26                 [-1, 64, 196, 196]        36,928
|    |    └─ReLU: 3-27                   [-1, 64, 196, 196]        --
├─Conv2d: 1-15                           [-1, 1, 196, 196]         65
==========================================================================================
Total params: 31,031,745
Trainable params: 31,031,745
Non-trainable params: 0
Total mult-adds (G): 43.59
==========================================================================================
Input size (MB): 0.75
Forward/backward pass size (MB): 239.89
Params size (MB): 118.38
Estimated Total Size (MB): 359.02
==========================================================================================
```
