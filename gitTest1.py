# master 작업
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class MNISTClassificationModel_FC_only(nn.Module):
    def __init__(self, n_classes):
        super(MNISTClassificationModel_FC_only, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),  # 입력 차원을 28*28*3으로 변경 (이미지가 RGB 3채널이므로)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),  # 출력 레이어의 노드 수를 n_classes로 변경
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
