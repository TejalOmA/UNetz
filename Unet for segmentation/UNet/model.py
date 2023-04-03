import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x

class down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = double_conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, input):
        x = self.conv(input)
        p = self.pool(x)

        return x, p

class up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = double_conv(out_channels+out_channels, out_channels)

    def forward(self, inputs, skip_connections):
        x = self.up(inputs)
        x = torch.cat([x, skip_connections], axis=1)
        x = self.conv(x)
        #print(x.shape)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        #Contracting Block
        self.d1 = down_block(3, 64)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)

        #Bottleneck
        self.b = double_conv(512, 1024)

        #Expanding Block
        self.u1 = up_block(1024, 512)
        self.u2 = up_block(512, 256)
        self.u3 = up_block(256, 128)
        self.u4 = up_block(128, 64)

        #Classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        #Contracting block
        c1, p1 = self.d1(inputs)
        c2, p2 = self.d2(p1)
        c3, p3 = self.d3(p2)
        c4, p4 = self.d4(p3)

        #Bottleneck
        b = self.b(p4)

        #Expanding block
        u1 = self.u1(b, c4)
        u2 = self.u2(u1, c3)
        u3 = self.u3(u2, c2)
        u4 = self.u4(u3, c1)
        #print(u4.shape)

        outputs = self.outputs(u4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    f = UNet()
    y = f(x)
    model = UNet()
    print(summary(model, input_size = (1, 3, 256, 256)))
