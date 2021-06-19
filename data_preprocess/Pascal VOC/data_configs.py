import yaml
import logging
config_file ='../../global_configs.yaml'
with open(config_file,'r') as yf:
    yaml_cfg = yaml.load(yf,Loader=yaml.FullLoader)
try:
    COCO_ROOT = yaml_cfg['DATA']['VOC']['VOC_FILE_ROOT']
except Exception as e:
    logging.error('config ERROR in root directory/global_configs.yaml !\t%s'%e.args)
    exit(-1)

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')