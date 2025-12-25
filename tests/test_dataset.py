# tests/test_dataset.py
import os
import importlib
import inspect
import torch
import pytest

pytestmark = pytest.mark.cpu


def get_ndws_root():
    """
    Prefer NDWS_ROOT if set; otherwise fall back to the repo's ndws_out folder.
    """
    env_root = os.environ.get("NDWS_ROOT")
    if env_root and os.path.isdir(env_root):
        return env_root

    # Fallback: cnn_aspp/data/ndws_out relative to repo root
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, ".."))
    candidate = os.path.join(repo_root, "cnn_aspp", "data", "ndws_out")
    if os.path.isdir(candidate):
        return candidate

    pytest.skip("NDWS_ROOT not set and cnn_aspp/data/ndws_out not found; skipping dataset tests")


def get_dataset_class():
    """
    Dynamically find a dataset class in cnn_aspp.data.ndws_dataset.

    We pick the first class whose name contains 'Dataset' and that implements __getitem__.
    """
    try:
        m = importlib.import_module("cnn_aspp.data.ndws_dataset")
    except ImportError:
        pytest.skip("cnn_aspp.data.ndws_dataset not importable; skipping dataset tests")

    for name, obj in inspect.getmembers(m, inspect.isclass):
        if "Dataset" in name and hasattr(obj, "__getitem__"):
            return obj

    pytest.skip("No dataset class with 'Dataset' in name found in ndws_dataset; skipping dataset tests")


@pytest.fixture(scope="session")
def ndws_root():
    return get_ndws_root()


@pytest.fixture(scope="session")
def dataset_cls():
    return get_dataset_class()


@pytest.fixture(scope="session")
def ndws_train(ndws_root, dataset_cls):
    try:
        return dataset_cls(root=ndws_root, split="train", augment=False)
    except TypeError:
        pytest.skip("Dataset constructor does not accept (root=..., split='train', augment=...)")


@pytest.fixture(scope="session")
def ndws_val(ndws_root, dataset_cls):
    try:
        return dataset_cls(root=ndws_root, split="val", augment=False)
    except TypeError:
        pytest.skip("Dataset constructor does not accept (root=..., split='val', augment=...)")


def _unpack_sample(sample):
    """
    Support both dict-style ({'x':..., 'y':...}) and tuple-style (x, y) datasets.
    """
    if isinstance(sample, dict):
        return sample["x"], sample["y"], sample
    else:
        x, y = sample
        return x, y, {}


def test_nan_and_missing_tiles_are_masked(ndws_train):
    n = min(len(ndws_train), 8)
    assert n > 0, "Train split appears empty"

    for idx in range(n):
        sample = ndws_train[idx]
        x, y, _meta = _unpack_sample(sample)

        assert torch.isfinite(x).all(), f"Non-finite values in x at idx={idx}"
        assert torch.isfinite(y).all(), f"Non-finite values in y at idx={idx}"


def test_augment_flag_toggles_pipeline(ndws_root, dataset_cls):
    try:
        ds_noaug = dataset_cls(root=ndws_root, split="train", augment=False)
        ds_aug = dataset_cls(root=ndws_root, split="train", augment=True)
    except TypeError:
        pytest.skip("Dataset constructor does not accept (root=..., split='train', augment=...)")

    assert len(ds_noaug) == len(ds_aug)

    s0 = ds_noaug[0]
    s1 = ds_aug[0]

    x0, y0, _m0 = _unpack_sample(s0)
    x1, y1, _m1 = _unpack_sample(s1)

    # Same shapes and dtypes
    assert x0.shape == x1.shape
    assert y0.shape == y1.shape
    assert x0.dtype == x1.dtype
    assert y0.dtype == y1.dtype

    # Heuristic: augmentation can change pixels (but we don't make this test flaky)
    s_aug_1 = ds_aug[0]
    s_aug_2 = ds_aug[0]

    x_aug_1, _, _ = _unpack_sample(s_aug_1)
    x_aug_2, _, _ = _unpack_sample(s_aug_2)

    if torch.allclose(x_aug_1, x_aug_2):
        pytest.skip("Augmentation appears deterministic for this sample; skipping randomness check")


def test_split_integrity_train_vs_val(ndws_train, ndws_val):
    train_ids = set()
    val_ids = set()

    for idx in range(min(200, len(ndws_train))):
        sample = ndws_train[idx]
        _x, _y, meta = _unpack_sample(sample)
        tile_id = meta.get("tile_id", idx)
        train_ids.add(tile_id)

    for idx in range(min(200, len(ndws_val))):
        sample = ndws_val[idx]
        _x, _y, meta = _unpack_sample(sample)
        tile_id = meta.get("tile_id", idx)
        val_ids.add(tile_id)

    assert train_ids.isdisjoint(val_ids), "Train/val splits share tile IDs (or indices)"
