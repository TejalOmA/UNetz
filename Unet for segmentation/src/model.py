import torch
import torch.nn as nn

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

class contracting_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = double_conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, input):
        x = self.conv(input)
        p = self.pool(x)

        return x, p

class expanding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = double_conv(out_channels+out_channels, out_channels)

    def forward(self, inputs, skip_connections):
        x = self.up(inputs)
        x = torch.cat([x, skip_connections], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        #Contracting block
        self.cb1 = contracting_block(3, 64)
        self.cb2 = contracting_block(64, 128)
        self.cb3 = contracting_block(128, 256)
        self.cb4 = contracting_block(256, 512)

        #Bottleneck
        self.b = double_conv(512, 1024)

        #Expanding block
        self.eb1 = expanding_block(1024, 512)
        self.eb2 = expanding_block(512, 256)
        self.eb3 = expanding_block(256, 128)
        self.eb4 = expanding_block(128, 64)

        #Classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        #Contracting block
        c1, p1 = self.cb1(inputs)
        c2, p2 = self.cb2(p1)
        c3, p3 = self.cb3(p2)
        c4, p4 = self.cb4(p3)

        #Bottleneck
        b = self.b(p4)

        #Expanding block
        eb1 = self.eb1(b, c4)
        eb2 = self.eb2(eb1, c3)
        eb3 = self.eb3(eb2, c2)
        eb4 = self.eb4(eb3, c1)

        outputs = self.outputs(eb4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)