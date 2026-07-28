"""Maintained mmBERT shadow inference and offline model development."""

from .core import MODEL_ID, MODEL_REVISION
from .inference import score_file

__all__ = ["MODEL_ID", "MODEL_REVISION", "score_file"]
