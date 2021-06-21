"""预处理COCO2014数据集的工具 COCO2014数据集可在github:https://github.com/albert-jin/CvT-SSD/blob/main/README.md Readme文件 找到并下载"""
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .data_configs import *


class COCODetection(Dataset):
    """
        coco_root:coco数据集目录
        image_set:coco下的文件夹,可选 trainval35k,train2014,val2014其中一个
    """

    def __init__(self, img_transform=None, coco_root=COCO_ROOT, image_set_name='trainval35k', dataset_name='MS COCO'):
        '''加载COCO从类别映射到下标的关系字典'''
        self.dataset_name = dataset_name
        label_flow = open(COCO_LABEL_FILE, 'r')
        self.img_transform = img_transform
        self.label_map = {int(label.split(',')[0]): int(label.split(',')[1]) for label in label_flow}
        self.image_directory = os.path.join(coco_root, IMAGES, image_set_name)
        # 将COCO的标注结果转成用torch.Tensor的锚框坐标和类别下标
        self.annotation = COCO(os.path.join(coco_root, ANNOTATIONS, 'instances_{}.json'.format(image_set_name)))
        self.ids = list(self.annotation.imgToAnns.keys())

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = '类: ' + self.__class__.__name__ + '\n'
        fmt_str += '    数据集名称: ' + self.dataset_name + '\n'
        fmt_str += '    图片实例数: {}\n'.format(self.__len__())
        fmt_str += '    图片所在文件夹: {}\n'.format(self.image_directory)
        fmt_str += '    Transforms (if any): {0}'.format(
            self.img_transform.__name__ if self.img_transform else "No") + '\n'
        fmt_str += '    Target Transforms (if any): ' + (
            self.coco_annotation_transform.__name__ if self.coco_annotation_transform else "No")
        return fmt_str

    def __getitem__(self, index):
        image, anno_target, _, _ = self.pull_item(index)
        return image, anno_target

    def coco_annotation_transform(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox) / scale)
                final_box.append(label_idx)
                res.append(final_box)  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                logging.error('no bbox in%s' % target)
        return res

    def pull_item(self, index):
        """给定下标,返回元组(图片,标注,高度,宽度) 标注是coco.loadAnns的返回"""
        img_id = self.ids[index]
        ann_ids = self.annotation.getAnnIds(imgIds=img_id)
        target = self.annotation.loadAnns(ann_ids)
        # 获取图片
        img_path = os.path.join(self.image_directory, self.annotation.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(img_path), "图片不存在:%s" % img_path
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        target = self.coco_annotation_transform(target, width, height)
        if self.img_transform:
            target = np.array(target)
            img, boxes, labels = self.img_transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]  # =>RGB
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        """给定下标,返回对应的图片"""
        img_id = self.ids[index]
        # 获取图片
        img_path = os.path.join(self.image_directory, self.annotation.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(img_path), "图片不存在:%s" % img_path
        return cv2.imread(img_path)

    def pull_anno(self, index):
        """给定下标,返回对应的标注 格式: [img_id, [(label1, bbox1-coords),(label2, bbox2-coords)...]] eg: ('001718', [('dog', (96, 13, 438, 332))])"""
        img_id = self.ids[index]
        ann_ids = self.annotation.getAnnIds(imgIds=img_id)
        return self.annotation.loadAnns(ann_ids)
