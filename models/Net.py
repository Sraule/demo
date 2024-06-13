import torch
import torch.nn as nn
import torchvision.models as models
import models.VGG.origin as vgg
from torch.nn import functional as F
import pyiqa

from options import opt


# 定义基础的3*3卷积单元
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with no padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块  通道数保持不变
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


class Depth_IQA(nn.Module):
    def __init__(self):
        super(Depth_IQA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 假设深度图是单通道的
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # 输出也是单通道的显著性图像

    def forward(self, d):
        d = F.relu(self.conv1(d))
        d = self.pool(d)
        d = F.relu(self.conv2(d))
        d = F.sigmoid(self.conv3(d))  # 使用Sigmoid激活函数来获取显著性图像
        return d


def compute_depth_weight(quality_score):
    return (1 - quality_score) + 0.5


# 对深度图质量的判断模块
class USODNet(nn.Module):
    def __init__(self):
        super(USODNet, self).__init__()

        ############### 定义RGB图像处理部分 ###############
        (
            self.encoder1_rgb,
            self.encoder2_rgb,
            self.encoder4_rgb,
            self.encoder8_rgb,
            self.encoder16_rgb,
        ) = vgg.Backbone_VGG16_in3()

        self.agant1 = self._make_agant_layer(16, 16)
        self.agant2 = self._make_agant_layer(16, 8)

        self.inplanes = 16
        self.deconv2a = self._make_transpose(TransBasicBlock, 16, 3, stride=2)
        self.inplanes = 8
        self.deconv2b = self._make_transpose(TransBasicBlock, 4, 3, stride=2)

        self.gate4 = GateBlock(512, 256)
        self.gate3 = GateBlock(1024, 512)
        self.gate2 = GateBlock(1024 + 256, 512)
        self.gate1 = GateBlock(1024 + 256 + 128, 512)
        self.gate0 = GateBlock(1024 + 256 + 128 + 64, 512)

        self.conv_rgb = nn.Conv2d(4, 1, kernel_size=3, padding=1)

        ############### 定义灰度深度图像处理部分 ###############
        self.score = pyiqa.create_metric('cnniqa').cuda()

        # channel == 1  -> repeat(1, 3, 1, 1)
        (
            self.encoder1_sg,
            self.encoder2_sg,
            self.encoder4_sg,
            self.encoder8_sg,
            self.encoder16_sg,
        ) = vgg.Backbone_VGG16_in3()

        self.conv_fu1 = conv1x1(1024, 256)
        self.conv_fu2 = conv1x1(512, 128)
        self.conv_fu3 = conv1x1(256, 64)

        self.sa_conv1 = Conv(32, 16)

        self.unet = UNet()

    def forward(self, image, sg, dep_type):

        # rgb encoder x0, x1, x2, x3, x4
        e0 = self.encoder1_rgb(image)
        e1 = self.encoder2_rgb(e0)
        e2 = self.encoder4_rgb(e1)
        e3 = self.encoder8_rgb(e2)
        e4 = self.encoder16_rgb(e3)

        if sg.size(1) == 1:
            sg = sg.repeat(1, 3, 1, 1)
        assert sg.size(1) == 1 or sg.size(1) == 3

        sg0 = self.encoder1_sg(sg)
        sg1 = self.encoder2_sg(sg0)
        sg2 = self.encoder4_sg(sg1)
        sg3 = self.encoder8_sg(sg2)
        sg4 = self.encoder16_sg(sg3)

        ff0_rgb_d = torch.cat([e0, sg0], dim=1)
        f0 = self.gcm0(ff0_rgb_d)
        ff1_rgb_d = torch.cat([e1, sg1], dim=1)
        f1 = self.gcm1(ff1_rgb_d)
        ff2_rgb_d = torch.cat([e2, sg2], dim=1)
        f2 = self.gcm2(ff2_rgb_d)
        ff3_rgb_d = torch.cat([e3, sg3], dim=1)
        f3 = self.gcm3(ff3_rgb_d)
        ff4_rgb_d = torch.cat([e4, sg4], dim=1)
        f4 = self.gcm4(ff4_rgb_d)

        gate_rgb = []
        gate4 = self.gate4(f4)
        gate_rgb.append(gate4)
        am4 = self.mixed_am4(f4, gate4)
        gate3 = self.gate3(f4,f3)
        gate_rgb.append(gate3)
        am3 = self.mixed_am3(f3, gate3)
        gate2 = self.gate2(f4, f3,f2)
        gate_rgb.append(gate2)
        am2 = self.mixed_am2(f2, gate2)
        gate1 = self.gate1(f4, f3, f2, f1)
        gate_rgb.append(gate1)
        am1 = self.mixed_am1(f1, gate1)
        gate0 = self.gate0(f4, f3, f2, f1, f0)
        gate_rgb.append(gate0)
        am0 = self.mixed_am0(f0, gate0)

        for i in range(5):
            gate_rgb[i] = F.interpolate(gate_rgb[i], size=(224, 224))

        ff1 = FURGB(am0, am1, am2)
        ff2 = FURGB(am1, am2, am3)
        ff3 = FURGB(am2, am3, am4)
        fusion_rgb = FURGB(ff1, ff2, ff3)
        conv_ff = nn.Conv2d(fusion_rgb.size(1), 16, kernel_size=1).cuda()
        fusion_rgb = conv_ff(fusion_rgb)

        fusion = self.agant1(fusion_rgb)
        fusion = self.deconv2a(fusion)
        fusion = self.agant2(fusion)
        fusion = self.deconv2b(fusion)
        u_net_f = fusion
        sal_rgb = self.conv_rgb(fusion)

        if dep_type == 'low':
            u_net_in = torch.cat([u_net_f, u_net_f, u_net_f], dim=1)  # [10, 12, 224, 224]
            sal = self.unet(u_net_in)
        else:
            depth_quality = compute_depth_weight(self.score(sg))
            sg_weight = compute_depth_weight(depth_quality)

            ff1_sg = FURGB(sg0, sg1, sg2)
            ff2_sg = FURGB(sg1, sg2, sg3)
            ff3_sg = FURGB(sg2, sg3, sg4)
            fusion_sg = FUSG(ff1_sg, ff2_sg, ff3_sg)

            sal_sg = self.conv_sg1(fusion_sg)
            u_net_d = sal_sg
            u_net_d = F.upsample(u_net_d, size=(224, 224))
            sg_weight = sg_weight.unsqueeze(-1).unsqueeze(-1)
            u_net_d = u_net_d * sg_weight
            sal_sg = self.conv_sg2(sal_sg)

            u_net_in = torch.cat((u_net_f, u_net_d), dim=1)     # [10, 12, 224, 224]
            sal = self.unet(u_net_in)
       
        assert dep_type in ['low', 'high']

        return sal, gate_rgb

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)




if __name__ == '__main__':
    torch.cuda.set_device(0)
    x = torch.rand(4, 3,  256, 256).cuda()
    l1 = torch.rand(4, 1, 256, 256).cuda()
    l2 = torch.rand(4, 1, 256, 256).cuda()
    model = USODNet().cuda()
    x, l_1 = model(x, l1)
    pass
