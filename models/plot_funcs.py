import torch
from matplotlib import cm
import matplotlib.colors as colors
import time

def get_color_map(num_classes):
    """
        Returns a function that maps each index in 0, 1,.. . N-1 to a distinct RGB color
    """
    color_norm = colors.Normalize(vmin=0, vmax=num_classes-1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='hsv')
    return [scalar_map.to_rgba(idx) for idx in range(num_classes)]

def show_images_bounding_boxes(plt, rgb_images, preds, labels, threshold =0.6):
    # Filter outputs with confidence scores lower than a threshold, we choose 60%
    """
        在matplotlib的rgb_image上添加模型预测的框框等信息
        plt: pyplot
        preds: outputs of model preds shapes: (1,21,200,5) batch_size,class_num,top_k, (1+4)
    """
    detections =preds.data
    batch_size = detections.shape[0]
    assert len(rgb_images) ==batch_size, "指定图片数和预测结果数不等!"
    num_classes= detections.shape[1]
    total_n = detections.shape[2]
    cmaps = get_color_map(num_classes)
    for batch_idx in range(batch_size):  # 针对batch中每个实例
        image_ =rgb_images[batch_idx]
        scale = torch.Tensor(image_.shape[1::-1]).repeat(2)  # 投射锚框与真实图片对应的尺度
        current_axis = plt.gca()
        for class_idx in range(0,num_classes):
            top_n =0
            while top_n< total_n:
                # +锚框
                score = detections[batch_idx, class_idx, top_n, 0]
                if score >= threshold:
                    pt = (detections[batch_idx, class_idx, top_n, 1:] * scale).cpu().numpy()
                    current_axis.add_patch(plt.Rectangle((pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1, fill=False, edgecolor=cmaps[class_idx], linewidth=2))
                    # +文字提示 (类别:置信度)
                    display_txt = '%s: %.3f' % (labels[class_idx-1], score)
                    current_axis.text(pt[0], pt[1], display_txt, bbox={'facecolor': cmaps[class_idx], 'alpha': 0.5})
                    top_n+=1
                else:
                    break
        plt.imshow(image_)
        plt.show()
        time.sleep(0.5)