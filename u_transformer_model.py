from torch import nn
from u_transformer_blocks import DoubleConv, Down, TransformerUp, OutConv
from u_transformer_attention import MHSA


class U_Transformer(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.transformer1 = MHSA(512, (28, 28, 18), (1, 1, 1), 512)
        self.up1 = TransformerUp(512, 256, (56, 56, 36), (28, 28, 18), 8)
        self.up2 = TransformerUp(256, 128, (112, 112, 72), (56, 56, 36), 8)
        self.up3 = TransformerUp(128, 64, (224, 224, 144), (112, 112, 72), 8)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        print("x1 size", x1.size())
        x2 = self.down1(x1)
        print("x2", x2.size())
        x3 = self.down2(x2)
        print("x3", x3.size())
        x4 = self.down3(x3)
        print("x4", x4.size())
        x4 = self.transformer1(x4)
        print("finished MHSA step")
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return x
