# tests/test_metrics.py
import torch
import pytest

pytestmark = pytest.mark.cpu


def binary_precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Simple binary metrics implementation for testing:
    y_true, y_pred are 1D tensors of 0/1.
    """
    y_true = y_true.int()
    y_pred = y_pred.int()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def test_binary_precision_recall_f1():
    # y_true: 1 1 1 0 0 0
    # y_pred: 1 0 1 1 0 0
    #
    # TP = 2  (positions 0, 2)
    # FP = 1  (position 3)
    # FN = 1  (position 1)
    #
    # precision = TP / (TP + FP) = 2/3
    # recall    = TP / (TP + FN) = 2/3
    # f1        = 2 * P * R / (P + R) = 2/3
    y_true = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.int64)
    y_pred = torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.int64)

    p, r, f1 = binary_precision_recall_f1(y_true, y_pred)

    assert pytest.approx(2 / 3, rel=1e-6) == p
    assert pytest.approx(2 / 3, rel=1e-6) == r
    assert pytest.approx(2 / 3, rel=1e-6) == f1
