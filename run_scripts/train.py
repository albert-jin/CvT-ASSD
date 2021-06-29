import os
import argparse
import yaml
import torch
import torch as t
import time
from torch.utils.data import DataLoader
from torch.backends import cudnn
from visdom import Visdom
import logging
import cv2
import numpy as np
from models.SSD.ssd_utils import MultiBoxLoss
from data_preprocess import COCODetection, VOCDetection
from utils import collect_fn, update_chart, output2maP, update_vali_chart
from data_preprocess.Pascal_VOC.data_configs import VOC_TEST_IMG_SETS, VOC_CLASSES
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from data_preprocess.utils import SSDAugmentation

# from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

GPU_ACCESS = t.cuda.is_available()
GPU_COUNTS = t.cuda.device_count()
GPU_DEVICES = [idx for idx in range(GPU_COUNTS)]

LOGGING_ITERS = 10
SAVING_ITERS = 10000
IMG_SIZE = 384


def get_args():
    # 指定dataset, 训练的iteration数(不是epoch), batch_size数, 是否用GPU(有GPU则默认用GPU)
    parser = argparse.ArgumentParser(description='Convolutional Transformer Based Single Shot MultiBox Detector '
                                                 'Training . Using VoC2017&2012 or COCO2014')
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'], type=str, help='dataset from VOC or COCO.')
    parser.add_argument('--network', default='CvT_SSD', choices=['CvT_SSD', 'CvT_ASSD', 'VGG_SSD', 'VGG_ASSD'],
                        type=str, help='network you wanna to train.')

    parser.add_argument('--use-gpu', default=GPU_ACCESS, choices=[True, False], type=bool,
                        help='whether use gpu to train.')
    # 如果其他显卡被占用,请在此指定使用的GPU卡序号, 默认使用机器上所有卡
    parser.add_argument('--gpu-ids', default=GPU_DEVICES, type=list, help='which gpus for training.')
    parser.add_argument('--base-network-configs-path', default=None, type=str, help='path of base-network to config, '
                                                                                    'you must give it when model-resume-path is None!')
    parser.add_argument('--base-network-resume-path', default=None, type=str, help='path of base-network to resume, '
                                                                                   'you must give it when model-resume-path is None!')
    parser.add_argument('--model-resume-path', default=None, type=str, help='path of checkpoint model file to resume '
                                                                            'training. if not, train from scratch')
    parser.add_argument('--model-save-path', default=os.path.abspath('./weights/'), type=str,
                        help='path of trained model directory')

    parser.add_argument('--batch-size', default=6, type=int, help='batch size for training')
    parser.add_argument('--iter-count', default=None, type=int,
                        help='total training iterations count default voc:12W,coco:40w')
    parser.add_argument('--start-iter-idx', default=0, type=int, help='start training iteration index')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optimizer')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    parser.add_argument('--visdom', default=False, type=bool, help='whether see running log in website.')
    parser.add_argument('--logger-path', default=os.path.abspath('./run_logs/'), type=str,
                        help='path of logging directory')

    return parser.parse_args()


def get_model(parser):
    # 获取模型参数配置
    model_name = parser.network
    MODEL_PATH = parser.model_resume_path
    BASE_MODEL_CONFIGS_PATH = parser.base_network_configs_path
    BASE_MODEL_FILE_PATH = parser.base_network_resume_path
    with open(BASE_MODEL_CONFIGS_PATH, 'r') as inp_:
        cvT_configs = yaml.load(inp_, Loader=yaml.FullLoader)
        cvT_model_configs = cvT_configs['MODEL']

    model = None
    num_classes = 21 if 'VOC' == parser.dataset else 81  # 获取物体类别送入网络
    if model_name == 'CvT_SSD':
        from models.CvT_SSD import build_ssd_from_cvt
        model = build_ssd_from_cvt(cvt_configs=cvT_model_configs,num_classes=num_classes,
                                   cvt_model_file_path=os.path.abspath(BASE_MODEL_FILE_PATH) if BASE_MODEL_FILE_PATH else None,
                                   model_path=os.path.abspath(MODEL_PATH) if MODEL_PATH else None)
    elif model_name == 'CvT_ASSD':
        from models.CvT_ASSD import build_assd_from_cvt
        model = build_assd_from_cvt(num_classes)
    elif model_name == 'VGG_SSD':
        from models.VGG_SSD import build_ssd_from_vgg
        model = build_ssd_from_vgg(mode='train', size=300, num_classes=21)
    elif model_name == 'VGG_ASSD':
        logging.error(f'I don`t build {model_name} model caused it`s no need now,if you wanna ,contact me by '
                      f'phone number: 13040617148.')
        exit(-1)
    else:
        logging.error('No such model named:{} !'.format(model_name))
        exit(-1)
    return model


def gpu_setting(parser):
    if parser.use_gpu and GPU_ACCESS:
        cudnn.enabled = True  # 允许使用非确定算法
        cudnn.benchmark = True  # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法, 优化运行效率
        cudnn.deterministic = False  # 使用非确定算法
        t.set_default_tensor_type('torch.cuda.FloatTensor')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(i for i in parser.gpu_ids)])
        # TODO: add torch distributed schedule if more than one gpu for training speed up. 这里可能还需优化.
    else:
        if GPU_ACCESS:
            logging.warning("WARNING: It looks like you have a CUDA device, but aren't using CUDA, run with --use-gpu "
                            "argument for training speed up.")
        t.set_default_tensor_type('torch.FloatTensor')


def train():
    # 训练准备
    parser = get_args()
    gpu_setting(parser)
    gpu_devices = [idx for idx, _ in enumerate(parser.gpu_ids)]
    GPU_TRAIN = parser.use_gpu and GPU_ACCESS
    PARAELLEL_FLAG = len(parser.gpu_ids) > 0 and GPU_TRAIN  # 是否分布式训练
    network = get_model(parser)
    optimizer = t.optim.SGD(network.parameters(), lr=parser.learning_rate, momentum=parser.momentum,
                            weight_decay=parser.weight_decay)
    num_classes = 21 if 'VOC' == parser.dataset else 81  # 获取物体类别送入网络
    iterations = 120000 if 'VOC' == parser.dataset else 400000
    lr_steps = (60000, 800000, 100000) if 'VOC' == parser.dataset else (250000, 300000, 350000)
    if parser.iter_count:
        iterations = parser.iter_count
    criterion = MultiBoxLoss(num_classes=num_classes, overlap_thresh=0.5, prior_for_matching=True, bkg_label=0,
                             neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False,
                             use_gpu=GPU_TRAIN)
    if PARAELLEL_FLAG:
        network = t.nn.DataParallel(network, device_ids=gpu_devices, output_device=gpu_devices[0])
        optimizer = t.nn.DataParallel(optimizer, device_ids=gpu_devices, output_device=gpu_devices[0])
    network.train()

    # 准备训练数据
    img_transform = SSDAugmentation(size=384)
    dataset = VOCDetection(img_transform=img_transform) if 'VOC' == parser.dataset else COCODetection(
        img_transform=img_transform)
    data_count = len(dataset)
    data_loader = DataLoader(dataset, batch_size=parser.batch_size, shuffle=True, collate_fn=collect_fn,
                             pin_memory=True, drop_last=True, generator=t.Generator(device=t.device('cuda')))  # num_workers=2
    data_iterator = iter(data_loader)

    # 准备测试数据
    test_dataset = VOCDetection(image_sets=VOC_TEST_IMG_SETS, dataset_name='VOC07_test')  # 测试集都用VOC2007_test
    test_data_count = len(test_dataset)
    test_data_loader = DataLoader(test_dataset, batch_size=parser.batch_size, shuffle=True,  # num_workers=2,
                                  collate_fn=collect_fn, pin_memory=True, generator=t.Generator(device=t.device('cuda')))
    test_data_iterator = iter(test_data_loader)

    if parser.visdom:
        vis_ = Visdom()
        vis_title = parser.network + ' training on ' + dataset.dataset_name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        vis_iter_opts = {'xlabel': 'Iteration', 'ylabel': 'Loss', 'title': vis_title, 'legend': vis_legend}
        vis_epoch_opts = {'xlabel': 'Iteration', 'ylabel': 'Loss', 'title': vis_title, 'legend': vis_legend}
        # 数据从iter1 & epoch1 开始
        iter_plot, epoch_plot = vis_.line(X=[0], Y=[[0, 0, 0]], opts=vis_iter_opts), vis_.line(X=[0], Y=[[0, 0, 0]],
                                                                                               opts=vis_epoch_opts)
        vis_vali_opts = {'xlabel': 'Iteration', 'ylabel': '(m)aP', 'title': vis_title,
                         'legend': VOC_CLASSES + ('maP(mean average precision)',)}
        vali_plot = vis_.line(X=[0], Y=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]],
                              opts=vis_vali_opts)  # 定时测试模型在VOC2007测试集上的maP指标
    # 日志文件记录 测试效果
    vali_logger = None
    if parser.logger_path:
        if os.path.isdir(parser.logger_path):
            log_name = '-'.join([parser.network, dataset.dataset_name, time.asctime().replace(' ', '_').replace(':', '_')])
            vali_logger = logging.getLogger()
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler_ = logging.FileHandler(os.path.join(parser.logger_path, log_name), encoding='UTF-8')
            handler_.setLevel(logging.INFO)
            handler_.setFormatter(formatter)
            vali_logger.addHandler(handler_)
            console_ = logging.StreamHandler()
            console_.setLevel(logging.INFO)
            console_.setFormatter(formatter)
            vali_logger.addHandler(console_)
        else:
            logging.error('--logger-path is not a directory, can`t save logs!')
            exit(-1)

    # 开始训练
    epoch_loc_loss = 0
    epoch_conf_loss = 0
    iter_loc_loss = 0
    iter_conf_loss = 0
    iters_count_per_epoch = data_count // parser.batch_size  # 每个epoch会遍历多少轮
    epoch_idx = (parser.start_iter_idx * parser.batch_size // data_count) + 1  # 通过 当前iteration与数据量大小计算epoch_idx
    test_metric = VOC07MApMetric(iou_thresh=0.5, class_names=VOC_CLASSES)
    epoch_start_time = time.time()
    iter_start_time = time.time()
    for iter_idx in range(parser.start_iter_idx, iterations):
        iter_idx_ = iter_idx + 1
        # 准备一个batch数据
        images, targets = next(data_iterator)
        if GPU_TRAIN:
            images = images.cuda()
            targets = [target.cuda() for target in targets]
        # 调整学习率
        if iter_idx_ in lr_steps:
            learning_rate_ = parser.learning_rate * (parser.gamma ** lr_steps.index(iter_idx_))
            for param in optimizer.param_groups:
                param['lr'] = learning_rate_
        ###########training##############
        output_ = network(images)
        loss_loc, loss_conf = criterion(output_, targets)
        epoch_loc_loss += loss_loc.data[0]
        epoch_conf_loss += loss_conf.data[0]
        iter_loc_loss += loss_loc.data[0]
        iter_conf_loss += loss_conf.data[0]
        loss_sum = loss_loc + loss_conf
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.module.step()  # optimizer.step()
        ############training#############
        # when iteration goes by per-LOGGING_ITERS.
        if iter_idx_ % LOGGING_ITERS == 0:
            iter_end_time = time.time()
            iter_spend_time = iter_end_time - iter_start_time
            vali_logger.info(
                'Iterations: [{0}]==[{1}], Epoch: [{2}], Speed: {3} pictures/sec, avg_iter_loc_loss: {4}, avg'
                '_iter_conf_loss: {5}.'.format(iter_idx_ - LOGGING_ITERS + 1, iter_idx_, epoch_idx,
                                               parser.batch_size
                                               * LOGGING_ITERS / iter_spend_time,
                                               iter_loc_loss / LOGGING_ITERS, iter_conf_loss / LOGGING_ITERS))
            iter_conf_loss = 0
            iter_loc_loss = 0

        if parser.visdom:
            update_chart(visdom=vis_, window_=iter_plot, step_idx=iter_idx_, loc_loss=loss_loc.data[0],
                         conf_loss=loss_conf.data[0])
        if iter_idx_ % iters_count_per_epoch == 0:  # when in a new epoch.
            # 添加当前epoch的loss汇总到 epoch_plot 图中
            avg_epoch_loc_loss = epoch_loc_loss / iters_count_per_epoch
            avg_epoch_conf_loss = epoch_conf_loss / iters_count_per_epoch
            if parser.visdom:
                update_chart(visdom=vis_, window_=epoch_plot, step_idx=epoch_idx, loc_loss=avg_epoch_loc_loss,
                             conf_loss=avg_epoch_conf_loss)
            epoch_end_time = time.time()
            epoch_spend_time = epoch_end_time - epoch_start_time
            epoch_start_time = epoch_end_time
            vali_logger.info(
                f"*** Epoch: [{epoch_idx}], time: [{epoch_spend_time}] sec, avg_epoch_loc_loss: {avg_epoch_loc_loss}, avg_epoch_conf_loss: {avg_epoch_conf_loss}")
            epoch_loc_loss = 0
            epoch_conf_loss = 0
            epoch_idx += 1
        if iter_idx_ % SAVING_ITERS == 0:
            # 测试当前模型在VOC2006_test上的maP.
            network.mode = 'test'
            test_metric.reset()
            for test_iter_idx in range(test_data_count // parser.batch_size):  # 迭代测试集一轮即可
                images, targets = next(test_data_iterator)
                with torch.no_grad():
                    output_ = network(images)
                locations_list, classes_list, scores_list = output2maP(output_)
                test_metric.update(pred_bboxes=locations_list, pred_labels=classes_list, pred_scores=scores_list,
                                   gt_bboxes=targets[:, :, :4], gt_labels=targets[:, :, 4], gt_difficults=None)
            aPs_maP_name, aPs_maP = test_metric.get()
            info_ = "\n".join(['\t' + k + " == " + str(v) for k, v in zip(aPs_maP_name, aPs_maP)])
            val_log_ = f'===[Iterations: {iter_idx_}, Validation: \n{info_}]'
            if vali_logger:
                vali_logger.info(val_log_)
            else:
                print(val_log_)
            update_vali_chart(vis_, window_=vali_plot, step_idx=iter_idx_, maPs=aPs_maP)
            network.mode = 'train'
            # 保存模型
            if os.path.isdir(parser.model_save_path):
                torch.save(network.module.state_dict(), os.path.join(parser.model_save_path, '_'.join(
                    [parser.network, dataset.dataset_name, 'iter' + str(iter_idx_)]) + '.pth'))
            else:
                vali_logger.error('--model-save-path is not a directory, can`t save model!')
                exit(-1)
    torch.save(network.module.state_dict(), os.path.join(parser.model_save_path, '_'.join(
        [parser.network, dataset.dataset_name, 'iter' + str(iterations)]) + '.pth'))
    vali_logger.info('finished !')


if __name__ == '__main__':
    train()
