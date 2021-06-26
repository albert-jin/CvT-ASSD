import torch as t
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = self.sum = self.avg = self.val = 0

    def reset(self):
        self.count = self.sum = self.avg = self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@t.no_grad()
def get_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res
