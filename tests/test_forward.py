import torch
import pytest

from cnn_aspp.models.aspp_tiny import ASPPTiny
from cnn_aspp.models.plain_cnn import PlainCNN


def make_tiny_aspp():
    # Match your NDWS channel count (12) explicitly
    return ASPPTiny(in_channels=12)


@pytest.mark.cpu
def test_aspp_forward_shape_and_dtype():
    model = make_tiny_aspp()
    model.eval()

    x = torch.randn(2, 12, 128, 128, dtype=torch.float32)
    out = model(x)

    # Expect [B, 1, H, W] logits
    assert out.shape[0] == 2            # batch
    assert out.shape[1] == 1            # channel
    assert out.shape[2:] == (128, 128)  # H, W
    assert out.dtype == torch.float32


@pytest.mark.cpu
def test_aspp_backward_gradient_flow():
    model = make_tiny_aspp()
    model.train()

    x = torch.randn(2, 12, 128, 128)
    out = model(x)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads, "Model has no trainable parameters"
    assert all(g is not None for g in grads), "Some params did not receive gradients"


@pytest.mark.cpu
def test_plain_cnn_forward_backward():
    # PlainCNN in your repo already has sensible defaults, so call it bare
    baseline = PlainCNN()
    baseline.train()

    x = torch.randn(2, 12, 128, 128)
    out = baseline(x)

    # Expect [B, 1, H, W] or at least same spatial resolution
    assert out.shape[0] == 2
    assert out.shape[2:] == (128, 128)

    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in baseline.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
