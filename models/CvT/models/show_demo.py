import os
import yaml
from cvt import *
from tqdm import tqdm
import time
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
from torch.nn import CrossEntropyLoss
from utils import AverageMeter, get_accuracy

MODEL_FILE_PATH = '../weights/CvT-w24-384x384-IN-22k.pth'  # CvT-13-224x224-IN-1k.pth
model_name =MODEL_FILE_PATH.split('/')[-1].split('.')[0]
DATA_CONFIGS_PATH = '../../../data_preprocess/data_configs.yaml'
MODEL_CONFIGS_PATH = '../configs/cvt-w24-384x384.yaml'
gpu_devices =[0, 1]

logger = logging.getLogger('test for CvT in imageNet-mini 3000+ pictures.')
log_dir_ ='./run_logs'
logger.setLevel(logging.INFO)
file_handle =logging.FileHandler(os.path.join(log_dir_,model_name+'_'+time.asctime().replace(' ','_').replace(':','_')+'_Validate.log'))
file_handle.setLevel(logging.NOTSET)
file_handle.setFormatter(logging.Formatter('%(levelno)s - %(asctime)s - %(message)s'))
logger.addHandler(file_handle)

try:
    # 获取训练测试数据地址
    with open(DATA_CONFIGS_PATH, 'r') as yf:
        yaml_cfg = yaml.load(yf, Loader=yaml.FullLoader)
    data_root = yaml_cfg['DATA']['IMAGE_NET_MINI']['FILE_ROOT']
    input_mean = yaml_cfg['DATA']['IMAGE_NET_MINI']['INPUT_MEAN']
    input_std = yaml_cfg['DATA']['IMAGE_NET_MINI']['INPUT_STD']
    # 获取模型参数配置
    with open(MODEL_CONFIGS_PATH, 'r') as inp_:
        configs = yaml.load(inp_, Loader=yaml.FullLoader)
    model_configs = configs['MODEL']
    image_size = configs['TEST']['IMAGE_SIZE']
    interpolation = configs['TEST']['INTERPOLATION']
except Exception as e:
    logging.error('config ERROR in VOC file in root directory/data_configs.yaml !\t%s' % e.args)
    exit(-1)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
cvt_network = ConvolutionVisionTransformer(model_configs=model_configs, mode='classifier', activate_method=QuickGELU,
                                           norm=LayerNorm_)
cvt_network.load_weights(pretrained_model_file=MODEL_FILE_PATH)
cvt_network.to(t.device('cuda'))
cvt_network = t.nn.DataParallel(cvt_network, device_ids=gpu_devices, output_device=gpu_devices[0])
cvt_network.eval()
transformer = transforms.Compose([transforms.Resize(int(image_size[0] / 0.875), interpolation=interpolation),
                                  transforms.CenterCrop(image_size[0]),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=input_mean, std=input_std)])
validate_loader = t.utils.data.DataLoader(
    dataset=datasets.ImageFolder(os.path.join(data_root, 'val'), transform=transformer),
    batch_size=2, shuffle=False, pin_memory=True,  # num_workers=1, persistent_workers=True,  # batch_size不能太大
    drop_last=False)
criterion = CrossEntropyLoss().cuda()
# 性能表现记录
loss_meter = AverageMeter()
top1_precision_meter = AverageMeter()
top5_precision_meter = AverageMeter()

start_tick = time.time()
logger.info(f'[{time.asctime()}]=> start validating...')
# Testing###########################
for batch_images, batch_labels in tqdm(validate_loader):
    iter_start_t = time.time()
    batch_images, batch_labels = batch_images.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True)
    with t.no_grad():
        outputs = cvt_network(batch_images)
        loss = criterion(outputs, batch_labels)
        loss_meter.update(loss.item(), batch_images.size(0))
        precision_1, precision_5 = get_accuracy(output=outputs, target=batch_labels, topk=(1, 5))
        top1_precision_meter.update(precision_1, batch_images.size(0))
        top5_precision_meter.update(precision_5, batch_images.size(0))
    iter_end_t = time.time()
    logger.info(f'\n[{time.asctime()}], Loss:{loss.item()}, Accuracy@1:{precision_1}, Accuracy@5:{precision_5}, '
                 f'duration-time:{"{:.5f}".format(iter_end_t-iter_start_t)}s.')
    iter_start_t = iter_end_t
# Testing###########################
logger.info(f'[{time.asctime()}]' + '=> finish validating. duration-time: {:.3f}s.'.format(time.time() - start_tick))
logger.info('=> Result (Average):\nLoss {:.3f}\nError@1 {:.3f}%\nError@5 {:.3f}%\nAccuracy@1 {:.3f}%\n'
             'Accuracy@5 {:.3f}%\t'.format(loss_meter.avg, 100-top1_precision_meter.avg, 100-top5_precision_meter.avg,
                                          top1_precision_meter.avg, top5_precision_meter.avg))

