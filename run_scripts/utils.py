import torch


def collect_fn(batch_data):
    # return images ,targets 指定batch数据的整理方式
    return torch.stack([img for img in batch_data]), list(map(lambda data: data[0], batch_data))


def update_chart(visdom,window_, step_idx, loc_loss, conf_loss):
    visdom.line(
        X=[[step_idx]*3],
        Y=[[loc_loss,conf_loss,conf_loss+loc_loss]],
        update='append',
        win =window_
    )
