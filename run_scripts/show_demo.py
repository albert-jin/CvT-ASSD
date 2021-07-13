"""
 test for the VGG_SSD model completing
 user can view the picture recognition performance .
"""
import os
import yaml
# import sys
# module_path = os.path.abspath('..')
# if module_path not in sys.path:
#     sys.path.append(module_path)  # 添加models环境变量 如果pycharm将models设为sources Root 则不需要.
import torch

from CvT_SSD import build_ssd_from_cvt

BASE_MODEL_CONFIGS_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                       '../models/CvT/configs/cvt-21-224x224.yaml')
with open(BASE_MODEL_CONFIGS_PATH, 'r') as inp_:
    cvT_configs = yaml.load(inp_, Loader=yaml.FullLoader)
    cvT_model_configs = cvT_configs['MODEL']

net_cvt_ssd = build_ssd_from_cvt(cvt_configs=cvT_model_configs, num_classes=21,
                                 model_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                         '../run_scripts/weights/CvT_SSD_VOC0712_iter120000.pth'),
                                 ssd_mode='test')
# print(net_vgg_ssd)

# showing demo picture from root directory.
import cv2
from matplotlib import pyplot as plt
from random import randint
import numpy as np
from plot_funcs import show_images_bounding_boxes
from data_preprocess import VOC_CLASSES, VOCDetection  # using voc dataset and show some train-val pictures.
plt.figure(figsize=(12, 12))

# (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

voc_loader = VOCDetection()
# picture_idxs = [randint(0, len(voc_loader) - 1) for _ in range(10)]
picture_idxs = [1, 2, 3]  # 车, 人马, 车


def img_transform(img):
    result = (cv2.resize(img, (224, 224)) - (104, 117, 123)).astype(np.float32)[:, :, ::-1].copy()
    return torch.from_numpy(result).permute(2, 0, 1)


pictures = [voc_loader.pull_image(idx) for idx in picture_idxs]
real_pictures = [cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (224, 224)) for picture in pictures]
inputs_ = [img_transform(picture) for picture in pictures]
inputs_ = torch.stack(inputs_)  # shapes:(batch_size,3,300,300)
if torch.cuda.is_available():
    inputs_ = inputs_.cuda()
outputs_ = net_cvt_ssd(inputs_)  # output (已经nms了) shape: [batch_size,num_classes,(每一个类中的前top_n个),(概率值+四个点==5)]
# 例如 (1,21,200,5)
show_images_bounding_boxes(plt, real_pictures, outputs_, VOC_CLASSES)
