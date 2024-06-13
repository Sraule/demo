import torch.nn as nn
from torch.nn import functional as F
import torch
class Depth_IQA(nn.Module):
    def __init__(self):
        super(Depth_IQA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 假设深度图是单通道的
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # 输出也是单通道的显著性图像
        self.sigmoid = nn.Sigmoid()

    def forward(self, d):
        d = F.relu(self.conv1(d))
        d = self.pool(d)
        d = F.relu(self.conv2(d))
        d = self.conv3(d)
        d = F.upsample(d, size=(224, 224))
        return self.sigmoid(d)
