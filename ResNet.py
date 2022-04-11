import torch
import torch.nn as nn

class BuildingBlock(nn.Module):
    expansion = 1 # 判断残差结构中主分支采用的卷积核的个数有没有发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x # 捷径分支的输出
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x

class BottleneckBolck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleneckBolck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channel=out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.merge_layer(block, 64, blocks_num[0])
        self.layer2 = self.merge_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self.merge_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self.merge_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def merge_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(channel * block.expansion)
                                       )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for i in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def ResNet18(num_classes):
    return ResNet(BuildingBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(BuildingBlock, [3, 4, 6, 3], num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(BottleneckBolck, [3, 4, 6, 3], num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(BottleneckBolck, [3, 4, 23 ,3], num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(BottleneckBolck, [3, 8, 36, 3], num_classes=num_classes)