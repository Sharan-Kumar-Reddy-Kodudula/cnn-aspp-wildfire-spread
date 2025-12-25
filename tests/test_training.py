# tests/test_training.py
import torch
import pytest

pytestmark = pytest.mark.cpu


class DummySegmentationDataset(torch.utils.data.Dataset):
    """
    Tiny synthetic dataset: mostly background with a few positive pixels.
    Just to test that a simple training loop reduces loss over a handful of steps.
    """

    def __init__(self, n=32, in_channels=12, h=64, w=64):
        self.n = n
        self.in_channels = in_channels
        self.h = h
        self.w = w

        g = torch.Generator().manual_seed(1337)
        self.x = torch.randn(n, in_channels, h, w, generator=g)
        self.y = (torch.rand(n, 1, h, w, generator=g) < 0.01).float()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


def make_model():
    from cnn_aspp.models.aspp_tiny import ASPPTiny
    return ASPPTiny(in_channels=12)


def test_micro_overfit_loss_drops():
    torch.manual_seed(1337)

    dataset = DummySegmentationDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = make_model()
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    step = 0
    max_steps = 20

    for epoch in range(3):
        for batch in loader:
            x = batch["x"]
            y = batch["y"]

            optimizer.zero_grad()
            logits = model(x)
            # Ensure shapes are compatible
            assert logits.shape == y.shape, f"logits shape {logits.shape}, target shape {y.shape}"

            loss = criterion(logits, y)
            assert torch.isfinite(loss)

            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach()))
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    assert len(losses) >= 5, "Not enough steps to observe loss trend"
    initial = sum(losses[:3]) / 3.0
    final = sum(losses[-3:]) / 3.0

    assert final < initial, f"Loss did not decrease: initial={initial}, final={final}"
