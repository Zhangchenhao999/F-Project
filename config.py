"""
Configuration settings for M-series Mac optimization
"""

# Device settings
MPS_MEMORY_FRACTION = 0.8  # Maximum fraction of GPU memory to use

# DataLoader settings
DATALOADER_CONFIG = {
    'num_workers': 2,  # Optimal for M-series
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 64,  # Adjusted for M-series GPU memory
    'grad_clip': 1.0,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
}

# Model settings
MODEL_CONFIG = {
    'dropout_rate': 0.2,
    'use_batch_norm': True,
} 