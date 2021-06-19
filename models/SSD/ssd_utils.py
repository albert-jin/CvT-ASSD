import torch
'''
    该文件旨在为SSD提供必要的工具函数
'''

def center2point(boxes):
    """
        将 boxes 映射到 左上右下两个点
    :param boxes: boxes shape:(boxCount,4) boxCount个边框,每个边框由[center_x,center_y,width,height]构成
    :return: shape:(boxCount,4) boxCount个边框,每个边框由[x1,y1,x2,y2]构成
    """
    return torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),1)

def point2center(boxes):
    '''
        和center2point 功能相反
    :param boxes: 略
    :return: 略
    '''
    return torch.cat(((boxes[:,2:]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]),1)

def intersection_of(boxes_a,boxes_b):
    '''
        计算两个框的交集区域大小
    :param box_a: 框的集合a
    :param box_b: 框的集合b
    :return:
    '''
    box_a_sum,box_b_sum =boxes_a.size(0) ,boxes_b.size(0)
    max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(box_a_sum, box_b_sum, 2),
                       boxes_b[:, 2:].unsqueeze(0).expand(box_a_sum, box_b_sum, 2))
    min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(box_a_sum, box_b_sum, 2),
                       boxes_b[:, :2].unsqueeze(0).expand(box_a_sum, box_b_sum, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return torch.reshape((inter[:, :, 0] * inter[:, :, 1]),(box_a_sum,box_b_sum))

def jaccard(boxes_a,boxes_b):
    '''
        计算两区域的jaccard重叠程度,计算公式:交集区域/并集区域, 介于0~1,0表示完全不重叠, 1 表示完全重叠
    :param boxes_a:
    :param boxes_b:
    :return:
    '''
    inter_area =intersection_of(boxes_a,boxes_b)
    area_a = ((boxes_a[:, 2]-boxes_a[:, 0]) *
              (boxes_a[:, 3]-boxes_a[:, 1])).unsqueeze(1).expand_as(inter_area)  # [A,B]
    area_b = ((boxes_b[:, 2]-boxes_b[:, 0]) *
              (boxes_b[:, 3]-boxes_b[:, 1])).unsqueeze(0).expand_as(inter_area)  # [A,B]
    union_area = area_a + area_b - inter_area
    return torch.reshape(inter_area / union_area,(boxes_a.size(0),boxes_b.size(0)))

def
