import logging
from itertools import product
from math import sqrt
import os
import torch
import yaml
from torch.nn import functional as F

CONFIG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../data_preprocess/data_configs.yaml')
try:
    with open(CONFIG_FILE, 'r') as yf:
        YAML_CONFIG = yaml.load(yf, Loader=yaml.FullLoader)
except Exception as e:
    logging.error('config ERROR in root directory/data_configs.yaml !\t%s' % e.args)
    exit(-1)

'''
    该文件旨在为SSD提供必要的工具函数
'''


class PriorBox:
    '''通过config得到一组候选框'''
    def __init__(self, PRIOR_BOX_CONFIG):
        """ PROIR_BOX_CONFIG从yaml读取全局配置中的候选框设置 """
        self.MIN_DIM = PRIOR_BOX_CONFIG['MIN_DIM']
        self.ASPECT_RATIOS = PRIOR_BOX_CONFIG['ASPECT_RATIOS']
        self.NUM_PRIOR = len(PRIOR_BOX_CONFIG['ASPECT_RATIOS'])
        self.VARIANCE = PRIOR_BOX_CONFIG['VARIANCE'] or [0.1]
        for v in self.VARIANCE:
            if v <= 0:
                raise ValueError('Variances must be greater than 0 !')
        self.FEATURE_MAPS = PRIOR_BOX_CONFIG['FEATURE_MAPS']
        self.MIN_SIZES = PRIOR_BOX_CONFIG['MIN_SIZES']
        self.MAX_SIZES = PRIOR_BOX_CONFIG['MAX_SIZES']
        self.CLIP = PRIOR_BOX_CONFIG['CLIP']
        self.STEPS = PRIOR_BOX_CONFIG['STEPS']

    def __call__(self):
        """ 调用并获取所有prior_boxes """
        with torch.no_grad():
            mean = []
            for k, f in enumerate(self.FEATURE_MAPS):
                for i, j in product(range(f), repeat=2):
                    f_k = self.MIN_DIM / self.STEPS[k]  # 每一步的像素值
                    cx = (j + 0.5) / f_k  # 除每步的像素值, cx是0~1之间的值
                    cy = (i + 0.5) / f_k
                    s_k = self.MIN_SIZES[k] / self.MIN_DIM
                    mean += [cx, cy, s_k, s_k]
                    s_k_prime = sqrt(s_k * (self.MAX_SIZES[k] / self.MIN_DIM))
                    mean += [cx, cy, s_k_prime, s_k_prime]
                    for ar in self.ASPECT_RATIOS[k]:
                        mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
            output = torch.Tensor(mean).view(-1, 4)
            if self.CLIP:
                output.clamp_(max=1, min=0)
            return output


def center2point(boxes):
    """
        将 boxes 映射到 左上右下两个点
    :param boxes: boxes shape:(boxCount,4) boxCount个边框,每个边框由[center_x,center_y,width,height]构成
    :return: shape:(boxCount,4) boxCount个边框,每个边框由[x1,y1,x2,y2]构成
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def point2center(boxes):
    '''
        和center2point 功能相反
    :param boxes: 略
    :return: 略
    '''
    return torch.cat(((boxes[:, 2:] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def intersection_of(boxes_a, boxes_b):
    '''
        计算两个框的交集区域大小
    :param box_a: 框的集合a
    :param box_b: 框的集合b
    :return:
    '''
    box_a_sum, box_b_sum = boxes_a.shape[0], boxes_b.shape[0]
    max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(box_a_sum, box_b_sum, 2),
                       boxes_b[:, 2:].unsqueeze(0).expand(box_a_sum, box_b_sum, 2))
    min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(box_a_sum, box_b_sum, 2),
                       boxes_b[:, :2].unsqueeze(0).expand(box_a_sum, box_b_sum, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return torch.reshape((inter[:, :, 0] * inter[:, :, 1]), (box_a_sum, box_b_sum))


def jaccard(boxes_a, boxes_b):
    '''
        计算两区域的jaccard重叠程度,计算公式:交集区域/并集区域, 介于0~1,0表示完全不重叠, 1 表示完全重叠
    :param boxes_a:
    :param boxes_b:
    :return:
    '''
    inter_area = intersection_of(boxes_a, boxes_b)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1).expand_as(inter_area)  # [A,B]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0).expand_as(inter_area)  # [A,B]
    union_area = area_a + area_b - inter_area
    return torch.reshape(inter_area / union_area, (boxes_a.shape[0], boxes_b.shape[0]))


def match(threshold, truth_boxes, priors_boxes, variances, labels, loc_t, conf_t, idx):
    """
        找出和 真实锚框 有较高(>threshold)重合度的priors_boxes(候选框),并编码bounding_boxes,并赋值与类别分和位置分
    :param threshold: 与truthbox的重叠度阈值 =>判断框内是否有物体的
    :param truth_boxes: 先验框框
    :param priors_boxes: 所有候选框
    :param variances:
    :param labels: 所有truth_boxes对应的物体类别
    :param loc_t:
    :param conf_t:
    :param idx:
    :return:
    """
    priors_boxes = center2point(priors_boxes)
    overlaps = jaccard(boxes_a=truth_boxes, boxes_b=priors_boxes)  # shape: (truth_boxes_num,priors_boxes_num)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # 找出和当前每个truth_box最重叠的priors_box
    best_prior_overlap.squeeze_(1)  # =>一维
    best_prior_idx.squeeze_(1)  # =>一维
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # 找出和当前每个priors_box最重叠的truth_box
    best_truth_overlap.squeeze_(0)  # =>一维
    best_truth_idx.squeeze_(0)  # =>一维
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # 将与truth_box最重叠的priors_box的重叠度变成2
    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j  # 将每个与truth_box最重叠的priors_box的里面的值变成对应的truth_box的下标
    matches = truth_boxes[best_truth_idx]  # 获取每个priors_box最重叠的truth_box shape:(priors_box_num,4)
    conf = labels[best_truth_idx] + 1  # 获取每个priors_box最可能的物体类别 Shape: (num_priors,)
    conf[best_truth_overlap < threshold] = 0  # 所有低于阈值的prior_box设为背景
    loc = encode(matches, priors_boxes, variances)  # 得到所有prior的位置损失值
    loc_t[idx] = loc  # 将损失(shape:[num_priors,4])写入第idx batch 位置损失中
    conf_t[idx] = conf  # 每个候选框对应的最可能物体类别 shape:(num_priors,1) 写入第idx batch中 类别损失中


def encode(matched, priors, variances):
    '''
        给定偏移预测variances,以及候选框和对应的真实宽,计算该偏移预测是否准确的损失
    :param matched: 真实框
    :param priors: 候选框
    :param variances: 预测的偏移值 (prior_bbox_num,2) like [(中心点偏移cx,cy),(长宽偏移cw,ch),..]
    :return: 位置得分
    '''
    cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    cxcy /= (variances[0] * priors[:, 2:])
    cwch = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    cwch = torch.log(cwch) / variances[1]
    return torch.cat([cxcy, cwch], 1)  # 位置得分


def decode(loc_loss, priors, variances):
    '''
        基于候选框,通过预测偏移量以及位置得分,计算出真实框
    :param loc_loss: 位置得分
    :param priors: 候选框
    :param variances: 预测偏移量
    :return: 真实框
    '''
    boxes = torch.cat((
        priors[:, :2] + loc_loss[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc_loss[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, max_overlap=0.5, top_k=200):
    '''
        非极大抑制 non-maximum suppression
        取置信分数最大的前top_k,继而选取top_k里所有两两之间overlap<max_overlap的出来
    :param boxes:
    :param scores:
    :param max_overlap:
    :param top_k:
    :return:
    '''
    keep = scores.new(scores.shape[0]).zero_().long()  # 先默认所有prior_box是背景下标
    if boxes.numel() == 0:  # 如果没有盒子,则都是背景
        return keep
    x1 = boxes[:, 0]  # 所有prior_box左上角x坐标
    y1 = boxes[:, 1]  # 所有prior_box左上角y坐标
    x2 = boxes[:, 2]  # 所有prior_box右下角x坐标
    y2 = boxes[:, 3]  # 所有prior_box右下角y坐标
    area = torch.mul(x2 - x1, y2 - y1)  # 所有prior_box的面积
    _, idx = scores.sort(0)  # 置信分从小到大的prior_box下标
    idx = idx[-top_k:]  # 置信分前top_k大的prior_box下标,最大的为[-1]
    count = 0
    while idx.numel() > 0:  # 对top_k个prior_box遍历
        i = idx[-1]  # 拿最后一个,最大分数的
        keep[count] = i  # 置信度最大的prior_box下标写到keep
        count += 1
        if idx.shape[0] == 1:
            break
        idx = idx[:-1]  # 删掉最后一个
        # 把分数第一以下的坐标写入
        xx1 =torch.index_select(x1, 0, idx)
        yy1 =torch.index_select(y1, 0, idx)
        xx2 =torch.index_select(x2, 0, idx)
        yy2 =torch.index_select(y2, 0, idx)
        # 截断长宽
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        # 候选框与当前最大置信框交叠的长宽 最小值:0
        inter_area = torch.clamp(xx2 - xx1, min=0.0) * torch.clamp(yy2 - yy1, min=0.0)
        other_areas = torch.index_select(area, 0, idx)  # 所有次候选框的面积
        union = (other_areas + area[i]) - inter_area
        IoU = inter_area / union  # 计算IoU
        idx = idx[IoU.le(max_overlap)]  # 只保留IoU不超过max_overlap的候选框
    return keep, count  # 置信度从高到低的候选框下标0~8372, 剩下框的个数


def log_sum_exp(conf_preds):
    """获取类别分"""
    conf_max = conf_preds.data.max()
    return torch.log(torch.sum(torch.exp(conf_preds - conf_max), 1, keepdim=True)) + conf_max


class L2Norm(torch.nn.Module):
    """L2归一化,使模型的输出重新恢复到合理的区间,特征更易表达"""

    def __init__(self, n_channels, gamma_scale, eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(self.n_channels))
        torch.nn.init.constant_(self.weight, gamma_scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x


class MultiBoxLoss(torch.nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.use_prior_for_matching = prior_for_matching
        self.background_label = bkg_label
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.encode_target = encode_target
        self.use_gpu = use_gpu
        dataset_name = 'VOC' if self.num_classes == 21 else "COCO"
        try:
            self.variance = YAML_CONFIG['DATA'][dataset_name]['PRIOR_BOX']['VARIANCE']
        except Exception as e:
            logging.error(
                f'error in  directory/data_configs.yaml DATA/{dataset_name}/PRIOR_BOX/VARIANCE !\t%s' % e.args)
            exit(-1)

    def forward(self, predictions, targets):
        """计算位置和类别的综合损失 L(x,c,l,g) = (Loss-conf(x, c) + α * Loss-loc(x,l,g)) / N"""
        loc_data, conf_data, priors = predictions
        num = loc_data.shape[0]
        priors = priors[:loc_data.shape[1], :]
        num_priors = (priors.shape[0])
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        loc_t = torch.autograd.Variable(loc_t, requires_grad=False)
        conf_t = torch.autograd.Variable(conf_t, requires_grad=False)
        pos = torch.gt(conf_t,0)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.shape[1] - 1)
        neg = torch.lt(idx_rank, num_neg.expand_as(idx_rank))
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class Detector(torch.nn.Module):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh):
        super(Detector, self).__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        dataset_name = 'VOC' if self.num_classes == 21 else "COCO"
        try:
            self.variance = YAML_CONFIG['DATA'][dataset_name]['PRIOR_BOX']['VARIANCE']
        except Exception as e:
            logging.error(
                f'error in  directory/data_configs.yaml DATA/{dataset_name}/PRIOR_BOX/VARIANCE !\t%s' % e.args)
            exit(-1)

    def forward(self, loc_data, conf_data, prior_data):
        """
                通过 loc & conf得分计算得出所有物体的真实框与类别
            loc_data: 每个priorBox位置偏移预测 Shape: [batch,num_priors,4]
            conf_data: 每个priorBox位置物体类别预测 Shape: [batch,num_priors,num_classes]
            prior_data: 固定的候选框们 Shape: [1,num_priors,4]
        """
        num = loc_data.shape[0]
        num_priors = prior_data.shape[0]
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,self.num_classes).transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.shape[0] == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        return output