import random, numpy as np
import math
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataloader import *
from MoFtE.MNIST.models import *

def set_seed(seed):
    torch.use_deterministic_algorithms(True)  # Ensures deterministic behavior
    torch.manual_seed(seed)  # Seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Seed for PyTorch (single GPU)
    torch.cuda.manual_seed_all(seed)  # Seed for PyTorch (all GPUs, if applicable)
    np.random.seed(seed)  # Seed for NumPy
    random.seed(seed)  # Seed for Python random
    # For compatibility with older PyTorch versions:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

