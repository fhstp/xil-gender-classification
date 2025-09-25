"""
CNN model factory - compatibility wrapper for architectures.py
"""

# Import everything from architectures for compatibility
from .architectures import *

# Ensure create_model is available
from .architectures import create_model

__all__ = ['create_model', 'GenderClassifier']