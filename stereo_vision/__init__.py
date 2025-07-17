"""
Stereo Vision Package

A Python package for stereo vision processing including feature extraction,
feature matching, and two-view geometry computations.
"""

from . import feature_extractor

__version__ = "0.1.0"
__author__ = "stereo-vision"

__all__ = [
    'feature_extractor',
    '__version__',
    '__author__'
]