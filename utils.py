import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
import numpy as np
from numpy import dot
from numpy.linalg import norm


def cossim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def hamming_sim(string1, string2): 
    # Start with a distance of zero, and count up
    # Start with a distance of zero, and count up
    res = string1==string2
    distance = np.sum(res) 
    # Return the final count of differences
    return distance/len(string1)

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 
def getRandPerson(exclude=0,totalcls=200):
    while True:
        indx =  np.random.randint(totalcls)
        if exclude != indx:
            return indx
        
def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)#, positive_label
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', remark='GCN'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, remark + '_model_best.pth.tar')

def visualize_graph(model, writer, input_size=(1, 3, 32, 32)):
    dummy_input = torch.rand(input_size)
    # with SummaryWriter(comment=name) as w:
    writer.add_graph(model, (dummy_input, ))

def get_parameters_size(model):
    total = 0
    for p in model.parameters():
        _size = 1
        for i in range(len(p.size())):
            _size *= p.size(i)
        total += _size
    return total