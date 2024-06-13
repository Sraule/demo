# -*- coding: utf-8 -*-
# @Time    : 2019
# @Author  : Lart Pang
# @FileName: VGG.py
# @GitHub  : https://github.com/lartpang

import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    no_grad = [ 'features.0.weight', 'features.0.bias',
                'features.1.weight', 'features.1.bias',
                'features.3.weight', 'features.3.bias',
                'features.4.weight', 'features.4.bias',
                'features.7.weight', 'features.7.bias',
                'features.8.weight', 'features.8.bias',
                'features.10.weight', 'features.10.bias',
                'features.11.weight', 'features.11.bias',
                'features.14.weight', 'features.14.bias',
                'features.15.weight', 'features.15.bias',
                'features.17.weight', 'features.17.bias',
                'features.18.weight', 'features.18.bias',
               'features.20.weight', 'features.20.bias',
               'features.21.weight', 'features.21.bias'
               # 'features.24.weight', 'features.24.bias',
               # 'features.25.weight', 'features.25.bias',
               # 'features.27.weight', 'features.27.bias',
               # 'features.28.weight', 'features.28.bias',
               # 'features.30.weight', 'features.30.bias',
               # 'features.31.weight', 'features.31.bias',
               # 'features.34.weight', 'features.34.bias',
               # 'features.35.weight', 'features.35.bias',
               # 'features.37.weight', 'features.37.bias',
               # 'features.38.weight', 'features.38.bias',
               # 'features.40.weight', 'features.40.bias',
               # 'features.41.weight', 'features.41.bias'
               ]
    # print(model)
    for name, value in model.named_parameters():
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
        # print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    for parameter in model.parameters():  # required_grad==False 才不会反向传播，只训练下面部分（微调）
        parameter.required_grad = False
    # for name, value in model.named_parameters():
    #     print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)


# if __name__ == "__main__":
#     div1, div2, div4, div8, div16 = models.VGG.origin.Backbone_VGG16_in3()
#     for model in models.VGG.origin.Backbone_VGG16_in3():
#         print([(name, params.size()) for name, params in model.named_parameters()])
