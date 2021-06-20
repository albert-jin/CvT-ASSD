"""预处理voc数据集的工具 voc数据集可在github:https://github.com/albert-jin/CvT-SSD/blob/main/README.md Readme文件 找到并下载"""
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.utils.data as data

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
from data_configs import *


class VOCDetection(data.Dataset):
    def __init__(self, img_transform=None, voc_root=VOC_ROOT, image_sets=VOC_IMG_SETS, dataset_name='VOC0712'):
        self.img_transform = img_transform
        self.dataset_name = dataset_name
        self.path_ids = []
        for (year, name) in image_sets:
            voc20xx_path = os.path.join(voc_root, 'VOC' + year)
            for line in open(os.path.join(voc20xx_path, 'ImageSets', 'Main', name + '.txt')):
                self.path_ids.append((voc20xx_path, line.strip()))
        self.annotation_path = os.path.join('%s', 'Annotations', '%s.xml')
        self.img_path = os.path.join('%s', 'JPEGImages', '%s.jpg')
        # 类别对应下标
        self.class2id = {y: x for x, y in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.path_ids)

    def __getitem__(self, index):
        """获取图像和标记数据"""
        return self.pull_item(index)[:2]

    def voc_annotation_transform(self, target, width, height, keep_difficult=False):
        """解析xml格式的标记数据 返回某张图片对应的标记xml文件里的物体类型和位置信息 [[x1, y1, x2, y2, label_index], ... ]"""
        result = []
        for obj in target.iter('object'):  # 遍历所有object节点
            difficult = int(obj.find('difficult').text) == 1
            if not keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()  # 获取物体类型
            bbox = obj.find('bndbox')  # 获取锚框坐标
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height  # 获取横纵坐标相对长高的比率
                bndbox.append(cur_pt)
            label_idx = self.class2id[name]  # 获取物体类型的下标id
            bndbox.append(label_idx)
            result += [bndbox]
        return result

    def pull_item(self, index):
        """获取第index个图片实例的所有信息(图像,标记,长,高)"""
        path_id = self.path_ids[index]
        xml_root = ET.parse(self.annotation_path % path_id).getroot()  # 获取第index实例的xml最顶层节点
        image = cv2.imread(self.img_path % path_id)  # 第index实例的图片
        height, width, channels = image.shape
        target = self.voc_annotation_transform(xml_root, width, height)  # 真实标签[位置的四个点, 类别下标]
        # 是否对图片, 标记, 类别进行另外处理.
        if self.img_transform:
            image, boxes, labels = self.img_transform(image, target[:, :4], target[:, 4])
            image = image[:, :, (2, 1, 0)]  # => RGB 还原图像格式
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))  # 还原标记格式
        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

    def pull_annotation(self, index):
        """返回第index的id以及其detect标记结果"""
        path_id = self.path_ids[index]
        annotation = ET.parse(self.annotation_path % path_id).getroot()
        target = self.voc_annotation_transform(annotation, 1, 1)
        return path_id[1], target

    def pull_image(self, index):
        """返回第index的图像数据"""
        path_id = self.path_ids[index]
        return cv2.imread(self.img_path % path_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        """返回第index的图像tensor"""
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
