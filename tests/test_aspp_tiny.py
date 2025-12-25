
# ---------------------------
# Tests
# File: tests/test_aspp_tiny.py
# ---------------------------
import torch
from cnn_aspp.models.aspp_tiny import ASPPTiny


def test_forward_shape():
    for C in (1, 3, 8):
        m = ASPPTiny(in_channels=C)
        x = torch.randn(2, C, 128, 128)
        y = m(x)
        assert y.shape == (2, 1, 128, 128)


def test_predict_range():
    m = ASPPTiny(in_channels=4)
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        p = m.predict(x)
    assert (0.0 <= p).all() and (p <= 1.0).all()

