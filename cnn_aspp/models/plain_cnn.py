import math
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class PlainCNN(nn.Module):
    """
    Light plain CNN for segmentation (no ASPP).
    3 ConvBlocks with BN+ReLU, then 1x1 head (1 logit channel).
    Head bias is set from a positive-class prior to avoid all-background collapse.
    """
    def __init__(self, in_ch: int = 12, base_ch: int = 64, pos_prior: float = 0.02):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 2

        self.block1 = ConvBlock(in_ch, c1)
        self.block2 = ConvBlock(c1, c2)
        self.block3 = ConvBlock(c2, c3)
        self.head   = nn.Conv2d(c3, 1, kernel_size=1)

        # He init for hidden convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.head:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

        # Head init: zero weights, bias = logit(pos_prior)
        prior = max(min(float(pos_prior), 0.99), 1e-3)
        logit = math.log(prior / (1.0 - prior))
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, logit)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)
