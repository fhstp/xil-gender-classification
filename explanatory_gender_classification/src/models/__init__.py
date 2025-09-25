"""Model architectures and utilities."""

from .architectures import GenderClassifier, create_model
from .rrr_model import RRRGenderClassifier
from .cnn_models import create_model  # Compatibility import

__all__ = [
    'GenderClassifier',
    'create_model', 
    'RRRGenderClassifier'
]