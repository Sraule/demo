import os
import torch
import torch.nn.functional as F
import sys

import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from math import exp
from torch.autograd import Variable
from options import opt

from models.Net import USODNet
from models.IQA.IQANet import Depth_IQA

from data import get_loader, test_dataset_bounds
from utils import clip_gradient, adjust_lr
from utils import IoU
import logging
import torch.backends.cudnn as cudnn

import pyiqa
obj = pyiqa.create_metric('cnniqa').cuda()

import torchvision
import cv2
import numpy as np

# 定义实施深度学习的设备
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# 对神经网络的训练过程
def train(train_loader, model, optimizer, epoch, save_path):
    global step  # 记录总循环次数
    model.train()
    loss_sal = 0
    loss_all = 0  # 定义总的loss值
    epoch_step = 0  # 记录当前epoch下的循环次数
    try:
        for i, (images, gts, depths_g) in enumerate(train_loader, start=1):
            loss_rgb = []
            loss_sg = []
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depths_g = depths_g.cuda()

            sal_dep = iqa_model(depths_g)
            dep_iou = IoU(sal_dep, gts)

            dep_type = decide_depth_quality(dep_iou)

            sal, gate_rgb = model(images, depths_g, dep_type)

            loss_sal = L1S(sal, gts)
            loss_rgb = 0
            for gate in gate_rgb:
                loss_rgb = loss_rgb + L1S(gate, gts)
            loss = loss_sal + loss_rgb

            loss.backward()  # 位置、前后关系请背过
            clip_gradient(optimizer, opt.clip)  # 对optimizer进行梯度裁剪    位置、前后关系请背过
            optimizer.step()
            step += 1  # 总循环次数+1  位置、前后关系请背过
            epoch_step += 1  # 当前epoch下的循环次数+1位   置、前后关系请背过
            loss_all += loss.data  # 记录总的loss值

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}'.format(datetime.now(),\
                        epoch, opt.epoch, i, total_step, loss.data))

        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'USOD_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'USOD_epoch_n+{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


if __name__ == '__main__':  # 主函数定义模型的主流程
    print("cuda is available:", torch.cuda.is_available())
    print("number of cuda:", torch.cuda.device_count())
    print("current device:", torch.cuda.current_device())
    print("CUDA Clear")
    torch.cuda.empty_cache()

    # 定义网络模型
    model = USODNet().cuda()
    iqa_model = Depth_IQA().cuda()
    iqa_model.load_state_dict(torch.load(r'./demo/models/IQA/pretrained/USOD_epoch_100.pth'))

    if (opt.load is not None):
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)  # 定义网络的优化器

    image_root = opt.rgb_root
    gt_root = opt.gt_root
    depth_root = opt.depth_root
    save_path = opt.save_path

    print('load data...')
    train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

    total_step = len(train_loader)

    L1S = torch.nn.SmoothL1Loss()
    step = 0  # 记录总循环次数z

    print("Start train...")
    for epoch in range(1, opt.epoch):
        print("current epoch:", epoch)
        cur_lr = adjust_lr(optimizer, epoch)
        print("current learning rate:", cur_lr)
        train(train_loader, model, optimizer, epoch, save_path)

    os.system("/usr/bin/shutdown")
