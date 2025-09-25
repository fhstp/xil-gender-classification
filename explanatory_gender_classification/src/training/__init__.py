"""Training utilities and trainers."""

# Import base trainer classes first
from .trainer import BaseTrainer, EarlyStopper

# Import specialized trainers
try:
    from .rrr_trainer import RRRTrainer
except ImportError:
    # Handle import error gracefully
    RRRTrainer = None

try:
    from .hybrid_trainer import HybridXILTrainer, run_hybrid_experiments
except ImportError:
    # Handle import error gracefully
    HybridXILTrainer = None
    run_hybrid_experiments = None

__all__ = [
    'BaseTrainer',
    'EarlyStopper',
]

# Add to __all__ only if successfully imported
if RRRTrainer is not None:
    __all__.append('RRRTrainer')
if HybridXILTrainer is not None:
    __all__.extend(['HybridXILTrainer', 'run_hybrid_experiments'])