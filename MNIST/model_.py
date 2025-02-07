import torch.nn as nn, torch
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5BatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5BatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d((6), affine=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d((16), affine=True)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.bn3 = nn.BatchNorm1d(120, affine=True)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84, affine=True)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

class LeNet5RMSNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5RMSNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.RMSNorm((6, 24, 24), elementwise_affine=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.RMSNorm((16, 8, 8), elementwise_affine=True)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.bn3 = nn.RMSNorm((120), elementwise_affine=True)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.RMSNorm((84), elementwise_affine=True)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

class LeNet5InstanceNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5InstanceNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.InstanceNorm2d((6), affine=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.InstanceNorm2d((16), affine=True)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.bn3 = nn.LayerNorm(120, elementwise_affine=True)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.LayerNorm(84, elementwise_affine=True)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x
