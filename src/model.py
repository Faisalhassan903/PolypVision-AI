import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU activations."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        features = self.conv(x)
        return features, self.pool(features)

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderStage(3, 32)
        self.enc2 = EncoderStage(32, 64)
        self.enc3 = EncoderStage(64, 128)
        self.enc4 = EncoderStage(128, 256)
        self.bottleneck = DoubleConv(256, 512)
    def forward(self, x):
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        return x1, x2, x3, x4, self.bottleneck(p4)

class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PolypUNet(nn.Module):
    """From-scratch U-Net used in the PolypVision exploratory study."""
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.dec4 = DecoderStage(512, 256, 256)
        self.dec3 = DecoderStage(256, 128, 128)
        self.dec2 = DecoderStage(128, 64, 64)
        self.dec1 = DecoderStage(64, 32, 32)
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, x):
        x1, x2, x3, x4, bottleneck = self.encoder(x)
        x = self.dec4(bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.output_conv(x)

if __name__ == '__main__':
    model = PolypUNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Output:', tuple(y.shape))
    print('Trainable parameters:', f'{params:,}')
