from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
import torch
from torch import nn


class CALayer(nn.Module):  # channel attention
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
#         print(self.pe[:, :, :x.size(2), :x.size(3)].size())
#         print(x.size())
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class Resent18(nn.Module):
    def __init__(self, classes=450):
        super(Resent18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)
        self.model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)

    def forward(self, x):
        x = self.model(x)
        return x

    
class Resent18hashing(nn.Module):
    def __init__(self, classes=450):
        super(Resent18hashing, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1024)
        self.model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.model.fc.weight.data.normal_(0, 0.01)
        self.model.fc.bias.data.fill_(0.0)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class Vgg16(nn.Module):
    def __init__(self, classes=450):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.fc = nn.Linear(self.model.classifier[6].in_features, classes)

    def forward(self, x):
        x = self.model(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y
    
def get_network_fn(name):
    networks_zoo = {
    'gcn': GCN(channel=4),
    'Resent18': Resent18(),
    'Resent18hashing': Resent18hashing(),
    'Vgg16': Vgg16(),
    'GCNCNN': GCNCNN(),
    'gcnca': GCNCA(),
    'gcn2way': GCN2way(),
    'gcn2wayca': GCN2wayCA(),
    'GCNPoe': GCNPoe(),
    'GCN2wayCAHashing': GCN2wayCAHashing(),
    'GCN2wayHashing': GCN2wayHashing(),
    'GCN2wayHashingsimple': GCN2wayHashingsimple(),
    'GCN2wayHashingsimple2': GCN2wayHashingsimple2(),
    'GCNhashingOneChannel': GCNhashingOneChannel(),
    }
    if name is '':
        raise ValueError('Specify the network to train. All networks available:{}'.format(networks_zoo.keys()))
    elif name not in networks_zoo:
        raise ValueError('Name of network unknown {}. All networks available:{}'.format(name, networks_zoo.keys()))
    return networks_zoo[name]