import torch


def collect_fn(batch_data):
    # return images ,targets 指定batch数据的整理方式
    return torch.stack([inst_[0] for inst_ in batch_data]), list(map(lambda inst_: torch.FloatTensor(inst_[1]), batch_data))


def update_chart(visdom,window_, step_idx, loc_loss, conf_loss):
    visdom.line(
        X=[[step_idx]*3],
        Y=[[loc_loss,conf_loss,conf_loss+loc_loss]],
        update='append',
        win =window_
    )

def update_vali_chart(visdom,window_, step_idx, maPs):
    visdom.line(
        X=[[step_idx]*len(maPs)],
        Y=[maPs],
        update='append',
        win =window_
    )

def output2maP(output_):
    """
    :param output_: SSD 模型在test模式下的输出
    shape: [batch_size,num_classes,(每一个类中的前top_n个),(概率值+四个点==5)]
    :return: 让VOC07MApMetric update的格式 output_
    """
    ot_ =output_.data.cpu().tolist()  # 转化为列表类型
    batch_size =ot_.shape[0]
    num_classes =ot_.shape[1]
    top_count =ot_.shape[2]
    classes_list, scores_list, locations_list=[], [], []
    for inst_ in range(batch_size):  # class_idx 背景是0,1~20是物体对应20类
        c_l, s_l, l_l=[], [], []
        for class_ in range(1,num_classes):  # class_ 还需要减一,mxnet上的类别是下标表示
            for top_ in range(top_count):
                c_l.append(class_-1)
                s_l.append(ot_[inst_,class_,top_,0])  # [inst_,class_,top_,:1]
                l_l.append(ot_[inst_,class_,top_,1:])
        classes_list.append(c_l)
        scores_list.append(s_l)
        locations_list.append(l_l)
    return locations_list, classes_list,scores_list  # 作为maP update的输入

