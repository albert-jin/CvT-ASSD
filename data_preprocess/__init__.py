"""
    集成了dataproprocess的工具包,处理数据集包括COCO2014 &VOC07_12
"""
import cv2
import numpy as np
import torch as t

from .COCO.coco import COCODetection, COCO_CLASSES
from .Pascal_VOC.voc import VOCDetection, VOC_CLASSES

detection_collate = lambda batch: (
t.stack([sample[0] for sample in batch]), [t.FloatTensor(sample[1]) for sample in batch])


class BaseTransform(object):
    """图像 =>尺寸统一&减去均值"""

    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return (cv2.resize(image, (self.size, self.size)).astype(np.float32) - self.mean).astype(np.float32)
