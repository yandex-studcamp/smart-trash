from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image
import torch


SpotterImageInput = str | Path | Image.Image | np.ndarray | torch.Tensor


class BaseSpotter(ABC):
    """Common inference interface for spotter predictors."""

    @abstractmethod
    def predict(self, image: SpotterImageInput) -> bool:
        """Return True when the image should be treated as anomalous."""

