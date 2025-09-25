"""
Configuration settings for the Explanatory Gender Classification project.
"""

import torch

class Config:
    # Training hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    SIZE = 224
    RANDOM_SEED = 42
    
    # Model settings
    NUM_CLASSES = 2
    MODEL_STORE_NAME = 'gender_classification.pt'
    
    # RRR specific parameters
    L2_GRADS = 1000  # Lambda for right reasons regularization
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths (update these for your setup)
    FEMALE_PATH = "data/female"
    MALE_PATH = "data/male"
    TRAIN_CSV = "data/train.csv"
    VAL_CSV = "data/val.csv"
    TEST_CSV = "data/test.csv"
    
    # Model architectures to support
    SUPPORTED_MODELS = [
        'densenet121', 
        'efficientnet_b0', 
        'googlenet', 
        'mobilenet_v2', 
        'resnet50', 
        'vgg16'
    ]
    
    # Early stopping parameters
    PATIENCE = 3
    MIN_DELTA = 10
    
    # Scheduler parameters
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.1