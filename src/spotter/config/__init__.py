"""Configuration helpers for the anomalib spotter."""

from .config import SpotterConfig, load_spotter_config

__all__ = ["SpotterConfig", "load_spotter_config"]
from .spotter_config import SpotterConfig, load_spotter_config, save_spotter_config

__all__ = ["SpotterConfig", "load_spotter_config", "save_spotter_config"]
