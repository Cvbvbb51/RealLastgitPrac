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


# freshman 작업
class MNISTClassificationModel_CNN(nn.Module):
    def __init__(self):
        super(MNISTClassificationModel_CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # 1=채널수, 32=출력채널(필터의 수), 필터크기 -> (32, H, W)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # output: (64, H, W)

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0
        )  # output: (H/2, W/2)

        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        # self.fc1 = nn.Linear(64 * 7 * 7, 64)

        self.fc2 = nn.Linear(64, 10)  # 출력 계층으로 바로 연결

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (32, 14, 14)
        # x = self.pool(torch.relu(self.conv2(x))) # (64, 7, 7)

        x = x.view(-1, 32 * 14 * 14)  # Flatten
        # x = x.view(-1, 64 * 7 * 7) # Flatten

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
