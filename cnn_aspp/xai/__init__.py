# cnn_aspp/xai/__init__.py

"""
XAI utilities for the wildfire CNN-ASPP project.

Currently includes:
- Grad-CAM wrappers
- CLI helpers for qualitative and quantitative analyses (Phase 8)
"""

from .gradcam import GradCAM, GradCAMOutput, upsample_to_match  # noqa: F401
