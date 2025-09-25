"""
Right-for-Right-Reasons (RRR) training utilities.
"""

import sys
import os

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.rrr_model import RRRTrainer, rrr_loss_function

# Export the RRRTrainer for compatibility
__all__ = ['RRRTrainer', 'rrr_loss_function']