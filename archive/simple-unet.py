# -------------------------
# Simple 3D UNet
# -------------------------
# class DoubleConv3D(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)


# class Down3D(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.MaxPool3d(kernel_size=2),
#             DoubleConv3D(in_ch, out_ch)
#         )

#     def forward(self, x):
#         return self.block(x)


# class Up3D(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.conv = DoubleConv3D(in_ch, out_ch)  # in_ch here is concatenated channels

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         return self.conv(x)


# class UNet3D(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.enc1 = DoubleConv3D(in_channels, 16)
#         self.enc2 = Down3D(16, 32)
#         self.enc3 = Down3D(32, 64)
#         self.enc4 = Down3D(64, 128)

#         self.center = Down3D(128, 256)

#         self.up4 = Up3D(256, 128)
#         self.up3 = Up3D(128, 64)
#         self.up2 = Up3D(64, 32)
#         self.up1 = Up3D(32, 16)

#         self.final = nn.Conv3d(16, num_classes, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         c = self.center(e4)
#         d4 = self.up4(c, e4)
#         d3 = self.up3(d4, e3)
#         d2 = self.up2(d3, e2)
#         d1 = self.up1(d2, e1)
#         return self.final(d1)