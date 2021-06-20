import yaml
import logging
config_file ='../../global_configs.yaml'
with open(config_file,'r') as yf:
    yaml_cfg = yaml.load(yf,Loader=yaml.FullLoader)
try:
    COCO_ROOT = yaml_cfg['DATA']['COCO']['FILE_ROOT']
    COCO_LABEL_FILE =yaml_cfg['DATA']['COCO']['COCO_LABELS_FILE']
except Exception as e:
    logging.error('config ERROR in COCO file in root directory/global_configs.yaml !\t%s'%e.args)
    exit(-1)

IMAGES ='images'
ANNOTATIONS ='annotations'
COCO_API ='PythonAPI'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')