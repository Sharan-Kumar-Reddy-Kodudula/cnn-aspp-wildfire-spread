from cnn_aspp.models.tiny_cnn import TinyCNN
import torch

def test_forward_shape():
    m = TinyCNN(in_ch=5)
    x = torch.randn(2,5,64,64)
    y = m(x)
    assert y.shape == (2,1,64,64)
