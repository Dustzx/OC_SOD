"""
OC_SOD - Object-Centric Salient Object Detection Framework
主要包含SAM2模型和视觉语言模型的集成
"""

__version__ = "0.1.0"
__author__ = "OC_SOD Contributors"

from .models import *
from .utils import *
from .data import *
from .core import *

__all__ = [
    'SAM2Model',
    'VisionLanguageModel',
    'Segmentor',
    'ImageDataset',
]
