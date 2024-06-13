import argparse
# 参数预定义的内容
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=151, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-04, help='learning rate')
parser.add_argument('--warmup', type=int, default=0, help='learning rate')
parser.add_argument('--scheduler', type=str, default='poly', help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping margin')                         # 定义控制梯度裁剪的参数
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate of learning rate')                # 定义learning decay的基本值   0.5
parser.add_argument('--decay_epoch', type=int, default=70, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--load_iqa', type=str, default='./demo/models/IQA/pretrained/SOD_epoch_100.pth', help='pretrained_IQANet')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--model_name', type=str, default=' Net', help='name of chosen model')

parser.add_argument('--rgb_root', type=str, default=r'./demo/RGBD_train/train_image//', help='the training rgb images root')
parser.add_argument('--gt_root',  type=str, default=r'./demo/RGBD_train/train_mask//', help='the training gt images root')
parser.add_argument('--depth_root',  type=str, default=r'./demo/RGBD_train/train_depth//', help='the training depth images root')
parser.add_argument('--test_rgb_root', type=str, default=r'./demo/NJU2K/test_image//', help='the test rgb images root')
parser.add_argument('--test_gt_root', type=str,  default=r'./demo/NJU2K/test_mask//', help='the test gt images root')
parser.add_argument('--test_depth_root',  type=str, default=r'./demo/NJU2K/test_depth//', help='the training gt images root')
parser.add_argument('--test_pred_root', type=str, default=r'./demo/models/result//', help='result image')
parser.add_argument('--save_path', type=str, default=r'./demo/models/BBSNet_cpts//', help='the path to save models and logs')
opt = parser.parse_args()
