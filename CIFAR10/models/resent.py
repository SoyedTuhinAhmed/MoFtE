import torch, torch.nn as nn, torch.nn.functional as F
import math
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation_type='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity = nn.Identity()
        self.adapter1 = Adapter(channel_dim=out_channels, reduction=4)
        self.adapter2 = Adapter(channel_dim=out_channels, reduction=4)

        # Define activations
        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act1 = gRReLU()
            self.act2 = gRReLU()
            self.act3 = gRReLU()

        # Downsample layer if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            if activation_type == 'relu':
                self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                         Adapter(channel_dim=out_channels, reduction=4),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    gRReLU(),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion),
                )

        self.stride = stride
        self.activation_type = activation_type

    def forward(self, x):
        identity = x

        # First convolution
        out = self.conv1(x)
        out = self.adapter1(out)
        if self.activation_type == 'grrelu':
            out = self.act1(out)  # Apply gRReLU right after Conv
        out = self.bn1(out)
        if self.activation_type == 'relu':
            out = self.act(out)  # Apply ReLU after BN

        # Second convolution
        out = self.conv2(out)
        out = self.adapter2(out)
        if self.activation_type == 'grrelu':
            out = self.act2(out)  # Apply gRReLU right after Conv
        out = self.bn2(out)

        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.identity(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation_type = 'relu'):
        super(Bottleneck, self).__init__()
        self.activation_type = activation_type

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if activation_type == 'relu':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif self.activation_type == 'grrelu':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    gRReLU(),
                    nn.BatchNorm2d(self.expansion*planes)
                )
        if activation_type == 'relu':
            self.conv1_act = nn.ReLU(inplace=True)
            self.conv2_act = nn.ReLU(inplace=True)
        elif self.activation_type == 'grrelu':
            self.conv1_act = gRReLU()
            self.conv2_act = gRReLU()
            self.conv3_act = gRReLU()

    def forward(self, x):
        if self.activation_type == 'relu':
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
        elif self.activation_type == 'grrelu':
            out = self.bn1(self.conv1_act(self.conv1(x)))
            out = self.bn2(self.conv2_act(self.conv2(out)))
            out = self.bn3(self.conv3_act(self.conv3(out)))

            out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation_type='relu'):
        super(ResNet, self).__init__()
        
        self.in_planes = 64
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.adapter1 = Adapter(channel_dim=64, reduction=32)
        self.bn1 = nn.BatchNorm2d(64)
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'grrelu':
            self.activation = gRReLU()
        elif activation_type == 'grrelu_pos':
            self.activation = gRReLU_pos()
        self.activation_type = activation_type
        self.flatten = nn.Flatten(start_dim=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation_type=activation_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation_type=activation_type)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation_type=activation_type)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation_type=activation_type)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        if activation_type != 'relu':
            self.bn = nn.BatchNorm1d(512*block.expansion)
            self.activation1 = gRReLU()
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, activation_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.adapter1(x)
        if self.activation_type == 'relu':
            x = self.bn1(x)
            x = self.activation(x)
        else:
            x = self.bn1(x)
            x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = self.flatten(x)
        if self.activation_type != 'relu':
            x = self.activation1(x)
            x = self.bn(x)
        x = self.fc(x)

        return x

def ResNet9(activation='relu'):
    return ResNet(BasicBlock, [1, 1, 1, 1], activation_type=activation)

def ResNet18(activation='relu'):
    return ResNet(BasicBlock, [2, 2, 2, 2], activation_type=activation)


def ResNet34(activation='relu'):
    return ResNet(BasicBlock, [3, 4, 6, 3], activation_type=activation)


def ResNet50(activation='relu'):
    return ResNet(Bottleneck, [3, 4, 6, 3], activation_type=activation)


def ResNet101(activation='relu'):
    return ResNet(Bottleneck, [3, 4, 23, 3], activation_type=activation)


def ResNet152(activation='relu'):
    return ResNet(Bottleneck, [3, 8, 36, 3], activation_type=activation)
