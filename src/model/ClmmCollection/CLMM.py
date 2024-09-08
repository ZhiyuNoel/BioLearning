import torchvision.models as models

import torch.nn as nn
import torch


# 18/34
class BasicBlock(nn.Module):
    expansion = 1  # 每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 50,101,152
class Bottleneck(nn.Module):
    expansion = 4  # 4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,  # 输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num):  # block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
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

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


"""
CLMM Network
"""


class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        # 使用预定义的 ResNet18，但不加载预训练权重
        self.cnn = models.resnet18(pretrained=False)

        # 修改输入的第一层卷积，使其适应单通道输入
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除 ResNet 最后的全连接层
        self.cnn.fc = nn.Identity()

        # 替换 layer4 以减少输出通道数
        self.cnn.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # 展平输出
        return x

class CLMM(nn.Module):
    def __init__(self, hidden_dim_img=128, hidden_dim_txt=32, num_layers=1):
        super(CLMM, self).__init__()
        self.hidden_dim_img = hidden_dim_img
        self.hidden_dim_txt = hidden_dim_txt
        self.num_layers = num_layers
        self.desc_dim = 3

        self.cnn = CustomResNet18()
        cnn_output_dim = 128

        # 视频帧 LSTM 部分
        self.lstm_video = nn.LSTM(cnn_output_dim, self.hidden_dim_img, self.num_layers, batch_first=True)

        # 文字描述 LSTM 部分
        self.txt_feature = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=27, kernel_size=5, padding=2),
            nn.BatchNorm1d(27),
            nn.ReLU(),
            nn.Conv1d(in_channels=27, out_channels=self.desc_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.desc_dim),
            nn.ReLU(),
        )
        self.lstm_text = nn.LSTM(self.desc_dim, self.hidden_dim_txt, self.num_layers, batch_first=True)

        # 特征融合和分类部分
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim_txt + self.hidden_dim_img, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # 可选：在全连接层加入 Dropout
            nn.Linear(64, 1)
        )

    def forward(self, x, desc):
        batch_size, timesteps, height, width = x.size()
        c_in = x.view(batch_size * timesteps, 1, height, width)

        # 处理视频帧
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out_video, _ = self.lstm_video(r_in)
        r_out_video = r_out_video[:, -1, :]  # 取最后一个时间步的输出

        # 处理文字描述
        desc = desc.transpose(1, 2)  # Transpose to (batch_size, features, timesteps)
        print(desc.shape)
        r_out_txt = self.txt_feature(desc)
        r_out_txt = r_out_txt.view(batch_size, timesteps, -1)
        r_out_txt, _ = self.lstm_text(r_out_txt)
        r_out_txt = r_out_txt[:, -1, :]  # 取最后一个时间步的输出

        # 特征融合
        combined = torch.cat((r_out_video, r_out_txt), dim=1)

        # 分类
        output = self.fc(combined)
        return output
