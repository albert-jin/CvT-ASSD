from SSD.ssd_model import SSD
from SSD.ssd_utils import L2Norm
from VGG.vgg_d import vgg_layers
import torch

VGG_NUM_CLASSES = 21


def build_ssd_from_vgg(mode='train', size=300):
    base_layers, extras_layers, loc_layers, conf_layers = vgg_layers(VGG_NUM_CLASSES)()
    model =SSD(base_layers, extras_layers, loc_layers, conf_layers, num_classes=VGG_NUM_CLASSES, mode=mode, size=size,
               l2Norm=L2Norm(512, 20), l2NormIdx=23)
    if torch.cuda.is_available():
        model = model.cuda()
    return model
