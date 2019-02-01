import time
import os
from torchvision import datasets, models, transforms
import copy
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.optim import lr_scheduler


#Used to visualize data
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = torch.Tensor(inp).numpy().transpose((1, 2, 0))
    mean = np.array([0.400, 0.445, 0.467], )
    std = np.array([0.2742, 0.259, 0.263])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated


def labelsToText():
    return { 0: "Outdoor", 1: "Indoor"}


