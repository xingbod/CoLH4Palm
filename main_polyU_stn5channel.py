from __future__ import division
import os
import time
import argparse
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from utils import accuracy, AverageMeter, save_checkpoint, visualize_graph, get_parameters_size
from torch.utils.tensorboard import SummaryWriter
from net_factory import get_network_fn



import numpy as np
# del test_loader
import tqdm
# test_loader

from loss import CSQLoss
from DsZoo import get_data

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats


parser = argparse.ArgumentParser(description='PyTorch GCN MNIST Training')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    metavar='N', help='GPU device ID (default: -1)')
parser.add_argument('--dataset_dir', default='../../MNIST', type=str, metavar='PATH',
                    help='path to dataset (default: ../MNIST)')
parser.add_argument('--comment', default='', type=str, metavar='INFO',
                    help='Extra description for tensorboard')
parser.add_argument('--model', default='gcn', type=str, metavar='NETWORK',
                    help='Network to train')

args = parser.parse_args()

use_cuda = (args.gpu >= 0) and torch.cuda.is_available()
writer = SummaryWriter(comment='_'+args.model+'_'+args.comment)
iteration = 0

def get_config():
    config = {
        "lambda": 0.5,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": 'GCNhashing5Channel',
        "dataset": "PolyU",
        "n_class":500,# pay attention
        "epoch": args.epochs,
        "test_map": 10,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0") if use_cuda else torch.device("cpu"),
        "bit_list": [1024],
    }
    return config


config = get_config()
print(config)
bit = 1024
device = config['device']

from net_factory import GCNSTNhashingOneChannel

# Load model
model = GCNSTNhashingOneChannel(inchannel=5)#GCNCNN
# print(model)

# Try to visulize the model
try:
	visualize_graph(model, writer, input_size=(1, 5, 128, 128))
except:
	print('\nNetwork Visualization Failed! But the training procedure continue.')

# Calculate the total parameters of the model

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['best_prec1'])
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))
train_loader, test_loader, num_train, num_test = get_data(config) 

# batch_size = 32
model = model.to(device)

optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
criterion = CSQLoss(config, bit)

# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=args.momentum, weight_decay=3e-05)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def test(net,test_loader):
    test_loss = AverageMeter()
    acc = AverageMeter()
    FEATS = []
    GT = []
    net.eval()
    features = {}
    with torch.no_grad():
        for batch_idx, img in enumerate(test_loader):
            rbn = img[0].permute(0, 3, 1, 2).to(device, dtype=torch.float)
            label = img[1].to(device)

            output = net(rbn)
            FEATS.append(output.cpu().numpy())
#             FEATS.append(features['feats'].cpu().numpy())
            GT.append(img[1].numpy())

    GCNFEATS = np.concatenate(FEATS)
    GT = np.concatenate(GT)
    GCNFEATS = normalized(GCNFEATS,1)
#     print('- feats shape:', GCNFEATS.shape)
#     print('- GT shape:', GT.shape)
    from numpy import dot
    from numpy.linalg import norm

    def cossim(a,b):
        return dot(a, b)/(norm(a)*norm(b))

    pred_scores = []
    gt_label = []

    for i in range(2000):
        for j in range(i+1,2000):
            # pred_scores.append(final[i,j].detach().cpu().numpy())
            a = cossim(GCNFEATS[i,:],GCNFEATS[j,:])
            pred_scores.append(a)
            gt_label.append(i//4 == j//4)

    pred_scores = np.array(pred_scores)
    gt_label = np.array(gt_label)

    Gen = pred_scores[gt_label]
    Imp = pred_scores[gt_label==False]
    Imp = Imp[np.random.permutation(len(Imp))[:len(Gen)]]


#     import seaborn as sns
#     sns.distplot(Gen,  kde=False, label='Gen')
#     # df =gapminder[gapminder.continent == 'Americas']
#     sns.distplot(Imp,  kde=False,label='Imp')
#     # Plot formatting
#     plt.legend(prop={'size': 12})
#     plt.title('Life Expectancy of Two Continents')
#     plt.xlabel('Life Exp (years)')
#     plt.ylabel('Density')

    # Calculating stats for classifier A
    stats_a = get_eer_stats(Gen, Imp)
    print(stats_a.eer)

    return stats_a.eer

##### INSPECT FEATURES


def train(epoch):
    current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
        config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")
    model.train()
    train_loss = 0
    for batch_idx, img in enumerate(train_loader):
        image = img[0].permute(0, 3, 1, 2).to(device, dtype=torch.float)
        label = img[1].to(device)

        optimizer.zero_grad()
        u = model(image)

        loss = criterion(u, label.float(), 0, config)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    train_loss = train_loss / len(train_loader)
    print("\b\b\b\b\b\b\b loss:%.5f, lr:%.5f" % (train_loss, optimizer.param_groups[0]['lr']))##loss:0.625
    scheduler.step()


Best_eer = 1.0
for epoch in range(args.start_epoch, args.epochs):
    print('------------------------------------------------------------------------')
    train(epoch+1)
    if epoch % 50 ==0:
        eer = test(model,test_loader)
        if eer < Best_eer:
            Best_eer = eer
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': Best_eer,
                'optimizer' : optimizer.state_dict(),
            }, True, filename='checkpoint'+config['net']+config['dataset']+'.pth.tar', remark=config['net']+config['dataset'])
        print("%s epoch:%d, bit:%d, dataset:%s,eer:%.5f, Best eer: %.5f" % (
            config["info"], epoch + 1, bit, config["dataset"], eer, Best_eer))


print('Finished!')
print('Best Best_eer:{:.2f}'.format(Best_eer))
writer.add_scalar('Best Best_eer', Best_eer, 0)
writer.close()