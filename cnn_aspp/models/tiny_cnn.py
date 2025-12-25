import math
import torch.nn as nn


class TinyCNN(nn.Module):
    """
    Minimal 3-layer CNN for segmentation with an optional dilation on convs
    and a head bias initialized from a foreground prior.

    Args:
        in_ch:     input channels
        mid:       hidden channels
        pos_prior: expected positive fraction (0..1) used to init head bias
        dilate:    dilation for conv1/conv2 (receptive field)
    """
    def __init__(self, in_ch: int = 12, mid: int = 128, pos_prior: float = 0.02, dilate: int = 1):
        super().__init__()
        pad = dilate
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=3, padding=pad, dilation=dilate)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=pad, dilation=dilate)
        self.head  = nn.Conv2d(mid, 1, kernel_size=1)
        self.act   = nn.ReLU(inplace=True)

        # Kaiming init for convs
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

        # Initialize head bias to logit(pos_prior) to avoid all-background collapse
        prior = max(min(float(pos_prior), 0.99), 1e-3)
        logit = math.log(prior / (1.0 - prior))
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, logit)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.head(x)
