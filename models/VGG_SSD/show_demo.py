"""
 test for the VGG_SSD model completing
 user can view the picture recognition performance .
"""
# import os
# import sys
# module_path = os.path.abspath('..')
# if module_path not in sys.path:
#     sys.path.append(module_path)  # 添加models环境变量 如果pycharm将models设为sources Root 则不需要.
import torch

from vgg_ssd import build_ssd_from_vgg

net_vgg_ssd = build_ssd_from_vgg(mode='test')
# print(net_vgg_ssd)
net_vgg_ssd.load_weights_from_old('./weights/ssd300_mAP_78.12.pth')

# showing demo picture from root directory.
import cv2
from matplotlib import pyplot as plt

plt.figure(figsize=(12, 12))
demo_picture_path = '../../data_preprocess/Pascal_VOC/demo_picture.jpg'
demo_picture = cv2.cvtColor(cv2.imread(demo_picture_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
plt.imshow(demo_picture)
plt.show()

# using voc dataset and show some train-val pictures.
from data_preprocess import VOCDetection
from random import randint
import numpy as np
from plot_funcs import show_images_bounding_boxes
from data_preprocess import VOC_CLASSES

voc_loader = VOCDetection()
picture_idxs = [randint(0, len(voc_loader) - 1) for _ in range(3)]


def img_transform(img):
    result = (cv2.resize(img, (300, 300)) - (104, 117, 123)).astype(np.float32)[:, :, ::-1].copy()
    return torch.from_numpy(result).permute(2, 0, 1)


pictures = [voc_loader.pull_image(idx) for idx in picture_idxs]
real_pictures = [cv2.cvtColor(picture, cv2.COLOR_BGR2RGB) for picture in pictures]
inputs_ = [img_transform(picture) for picture in pictures]
inputs_ = torch.stack(inputs_)  # shapes:(batch_size,3,300,300)
if torch.cuda.is_available():
    inputs_ = inputs_.cuda()
outputs_ = net_vgg_ssd(inputs_)
show_images_bounding_boxes(plt, real_pictures, outputs_, VOC_CLASSES)
