from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GConv
from torchvision import datasets, models, transforms

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


class GCN(nn.Module):
    def __init__(self, channel=4):
        super(GCN, self).__init__()
        self.channel = channel
        self.model = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),

            # GConv(80, 160, 5, padding=0, stride=1, M=channel, nScale=5, bias=False),
            # nn.BatchNorm2d(160*channel),
            # nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(80, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        x = self.model(x)
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        x = x.view(-1, 80, self.channel)
#         print('**x.view()',x.size())
        x = torch.max(x, 2)[0]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GCN2way(nn.Module):
    def __init__(self, channel=4):
        super(GCN2way, self).__init__()
        self.channel = channel
        self.model_vein = nn.Sequential(
            GConv(2, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        self.model_print = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        
        self.fc1 = nn.Linear(80*2, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        prints = x[:,:3,:,:]
        veins = x[:,3:,:,:]
#         print('**prints x.size()',prints.size())
#         print('**veins x.size()',veins.size())
#         print('.dtype prints',prints.dtype)
#         print('.dtype veins',veins.dtype)
        xprints = self.model_print(prints)#RGB img 
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        xprints = xprints.view(-1, 80, self.channel)
#         print('**x.view()',x.size())
        xprints = torch.max(xprints, 2)[0]
    
        xveins = self.model_vein(veins)#RGB img 
        xveins = xveins.view(-1, 80, self.channel)
        xveins = torch.max(xveins, 2)[0]
        
        x = torch.cat([xprints, xveins], 1)
    
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
import math
import torch
from torch import nn


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


class GCN2wayCA(nn.Module):
    def __init__(self, channel=4):
        super(GCN2wayCA, self).__init__()
        self.channel = channel
        self.pos_encoding = PositionEncodingSine(
            5,
            temp_bug_fix=False)
        self.model_vein = nn.Sequential(
            GConv(2, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            CALayer(10*channel),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            CALayer(20*channel),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            CALayer(40*channel),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            CALayer(80*channel),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        self.model_print = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            CALayer(10*channel),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            CALayer(20*channel),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            CALayer(40*channel),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            CALayer(80*channel),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        
        self.fc1 = nn.Linear(80*2, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        x = self.pos_encoding(x)

        prints = x[:,:3,:,:]
        veins = x[:,3:,:,:]
        ## xb addd pos enbeeding
#         print('**prints x.size()',prints.size())
#         print('**veins x.size()',veins.size())
#         print('.dtype prints',prints.dtype)
#         print('.dtype veins',veins.dtype)
        xprints = self.model_print(prints)#RGB img 
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        xprints = xprints.view(-1, 80, self.channel)
#         print('**x.view()',x.size())
        xprints = torch.max(xprints, 2)[0]
    
        xveins = self.model_vein(veins)#RGB img 
        xveins = xveins.view(-1, 80, self.channel)
        xveins = torch.max(xveins, 2)[0]
        
        feat = torch.cat([xprints, xveins], 1)
        
        x = self.fc1(feat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x,feat



class GCNPoe(nn.Module):
    def __init__(self, channel=4):
        super(GCNPoe, self).__init__()
        self.channel = channel
        self.pos_encoding = PositionEncodingSine(
            3,
            temp_bug_fix=False)
        self.model = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),

            # GConv(80, 160, 5, padding=0, stride=1, M=channel, nScale=5, bias=False),
            # nn.BatchNorm2d(160*channel),
            # nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(80, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
        print('**input x.size()',x.size())
        x = self.pos_encoding(x)
        print('**pos_encoding x.size()',x.size())

        x = self.model(x)
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        x = x.view(-1, 80, self.channel)
#         print('**x.view()',x.size())
        x = torch.max(x, 2)[0]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class GCN2wayCAHashing(nn.Module):
    def __init__(self, channel=4):
        super(GCN2wayCAHashing, self).__init__()
        self.channel = channel
        self.pos_encoding = PositionEncodingSine(
            5,
            temp_bug_fix=False)
        self.model_vein = nn.Sequential(
            GConv(2, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            CALayer(10*channel),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            CALayer(20*channel),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            CALayer(40*channel),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            CALayer(80*channel),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        self.model_print = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            CALayer(10*channel),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            CALayer(20*channel),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            CALayer(40*channel),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            CALayer(80*channel),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        
        self.hash_layer = nn.Linear(80*2, 1024)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        
#         self.fc1 = nn.Linear(80*2, 1024)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        x = self.pos_encoding(x)
        prints = x[:,:3,:,:]
        veins = x[:,3:,:,:]
#         print('**prints x.size()',prints.size())
#         print('**veins x.size()',veins.size())
#         print('.dtype prints',prints.dtype)
#         print('.dtype veins',veins.dtype)
        xprints = self.model_print(prints)#RGB img 
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        xprints = xprints.view(-1, 80, self.channel)
#         print('**x.view()',x.size())
        xprints = torch.max(xprints, 2)[0]
    
        xveins = self.model_vein(veins)#RGB img 
        xveins = xveins.view(-1, 80, self.channel)
        xveins = torch.max(xveins, 2)[0]
        
        feat = torch.cat([xprints, xveins], 1)
        
        y = self.hash_layer(feat)
        return y
    

class GCN2wayHashing(nn.Module):
    def __init__(self, channel=4):
        super(GCN2wayHashing, self).__init__()
        self.channel = channel
#         self.pos_encoding = PositionEncodingSine(
#             10*channel,
#             temp_bug_fix=False)
        self.model_vein = nn.Sequential(
            GConv(2, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        self.model_print = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        
        self.hash_layer = nn.Linear(80*2, 1024)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        
#         self.fc1 = nn.Linear(80*2, 1024)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        prints = x[:,:3,:,:]
        veins = x[:,3:,:,:]
#         print('**prints x.size()',prints.size())
#         print('**veins x.size()',veins.size())
#         print('.dtype prints',prints.dtype)
#         print('.dtype veins',veins.dtype)
        xprints = self.model_print(prints)#RGB img 
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        xprints = xprints.view(-1, 80, self.channel)
#         print('**x.view()',x.size())
        xprints = torch.max(xprints, 2)[0]
    
        xveins = self.model_vein(veins)#RGB img 
        xveins = xveins.view(-1, 80, self.channel)
        xveins = torch.max(xveins, 2)[0]
        
        feat = torch.cat([xprints, xveins], 1)
        
        y = self.hash_layer(feat)
        return y




class GCNhashingOneChannel(nn.Module):
    def __init__(self, channel=4):
        super(GCNhashingOneChannel, self).__init__()
        self.channel = channel
        self.model = nn.Sequential(
            GConv(1, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),

        )
        self.hash_layer = nn.Linear(320*12*12, 1024)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
#         print('**input x.size()',x.size())
        x = self.model(x)
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        x = x.view(-1, 320*12*12)
#         print('**x.view()',x.size())
#         x = torch.max(x, 2)[0]
        x = self.hash_layer(x)

        return x


    
class GCN2wayHashingsimple(nn.Module):
    def __init__(self, channel=4):
        super(GCN2wayHashingsimple, self).__init__()
        self.channel = channel
#         self.pos_encoding = PositionEncodingSine(
#             10*channel,
#             temp_bug_fix=False)
        self.model_vein = nn.Sequential(
            GConv(2, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        self.model_print = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        
        self.hash_layer = nn.Linear(320*12*12*2, 1024)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        
#         self.fc1 = nn.Linear(80*2, 1024)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        prints = x[:,:3,:,:]
        veins = x[:,3:,:,:]
#         print('**prints x.size()',prints.size())
#         print('**veins x.size()',veins.size())
#         print('.dtype prints',prints.dtype)
#         print('.dtype veins',veins.dtype)
        xprints = self.model_print(prints)#RGB img 
#         print('**xprints.size()',xprints.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        xprints = xprints.view(-1, 320*12*12)
#         print('**x.view()',x.size())
#         xprints = torch.max(xprints, 2)[0]
    
        xveins = self.model_vein(veins)#RGB img 
        xveins = xveins.view(-1, 320*12*12)
#         xveins = torch.max(xveins, 2)[0]
        
        feat = torch.cat([xprints, xveins], 1)
        
        y = self.hash_layer(feat)
        return y


    
class GCN2wayHashingsimple2(nn.Module):
    def __init__(self, channel=4):
        super(GCN2wayHashingsimple2, self).__init__()
        self.channel = channel
#         self.pos_encoding = PositionEncodingSine(
#             10*channel,
#             temp_bug_fix=False)
        self.model_vein = nn.Sequential(
            GConv(2, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            PositionEncodingSine(20*channel,temp_bug_fix=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        self.model_print = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            PositionEncodingSine(10*channel,temp_bug_fix=False),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            PositionEncodingSine(20*channel,temp_bug_fix=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),
        )
        
        self.hash_layer = nn.Linear(320*12*12*2, 1024)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        
#         self.fc1 = nn.Linear(80*2, 1024)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        prints = x[:,:3,:,:]
        veins = x[:,3:,:,:]
#         print('**prints x.size()',prints.size())
#         print('**veins x.size()',veins.size())
#         print('.dtype prints',prints.dtype)
#         print('.dtype veins',veins.dtype)
        xprints = self.model_print(prints)#RGB img 
#         print('**xprints.size()',xprints.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        xprints = xprints.view(-1, 320*12*12)
#         print('**x.view()',x.size())
#         xprints = torch.max(xprints, 2)[0]
    
        xveins = self.model_vein(veins)#RGB img 
        xveins = xveins.view(-1, 320*12*12)
#         xveins = torch.max(xveins, 2)[0]
        
        feat = torch.cat([xprints, xveins], 1)
        
        y = self.hash_layer(feat)
        return y

        
class GCNCA(nn.Module):
    def __init__(self, channel=4):
        super(GCNCA, self).__init__()
        self.channel = channel
        self.model = nn.Sequential(
            GConv(3, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
#             CALayer(10*channel),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
#             CALayer(20*channel),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
#             CALayer(40*channel),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
#             CALayer(80*channel),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),

#             GConv(80, 160, 5, padding=0, stride=1, M=channel, nScale=5, bias=False),
#             nn.BatchNorm2d(160*channel),
#             nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(160, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
#         print('**input x.size()',x.size())
        x = self.model(x)
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        x = x.view(-1, 160, self.channel)
#         print('**x.view()',x.size())
        x = torch.max(x, 2)[0]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GCNCNN(nn.Module):
    def __init__(self, channel=4):
        super(GCNCNN, self).__init__()
        self.channel = channel
        self.model = nn.Sequential(
            GConv(5, 10, 5, padding=0, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(10*channel),
            nn.ReLU(inplace=True),

            GConv(10, 20, 5, padding=0, stride=2, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(20, 40, 5, padding=0, stride=2, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(40*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            GConv(40, 80, 5, padding=0, stride=2, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80*channel),
            nn.ReLU(inplace=True),

            # GConv(80, 160, 5, padding=0, stride=1, M=channel, nScale=5, bias=False),
            # nn.BatchNorm2d(160*channel),
            # nn.ReLU(inplace=True),
        )
        
        self.conv1_1 = nn.Conv2d(5, 8, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.fc1 = nn.Linear(4176, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 450)

    def forward(self, x):
        orig = x
        
        x = self.model(x)
#         print('**x.size()',x.size())
        # x = x.view(-1, self.channel, 80)
        # x = torch.max(x, 1)[0]
        # x = x.view(-1, 80 * self.channel)
        x = x.view(-1, 80, self.channel)
        # print('**x.view()',x.size())
        x = torch.max(x, 2)[0]
        
        cnn_conv1 = self.lrelu(self.conv1_1(orig))
#         cnn_conv1 = self.lrelu(self.conv1_2(cnn_conv1))
        cnn_pool1 = self.pool1(cnn_conv1)

        cnn_conv2 = self.lrelu(self.conv2_1(cnn_pool1))
#         cnn_conv2 = self.lrelu(self.conv2_2(cnn_conv2))
        cnn_pool2 = self.pool1(cnn_conv2)

        cnn_conv3 = self.lrelu(self.conv3_1(cnn_pool2))
#         cnn_conv3 = self.lrelu(self.conv3_2(cnn_conv3))
        cnn_pool3 = self.lrelu(self.pool1(cnn_conv3))

        cnn_conv4 = self.lrelu(self.conv4_1(cnn_pool3))
#         cnn_conv4 = self.lrelu(self.conv4_2(cnn_conv4))
        cnn_pool4 = self.lrelu(self.pool1(cnn_conv4))

#         print('cnn_pool4',cnn_pool4.size())
#         print('x',x.size())
        cnn_out = cnn_pool4.contiguous().view(-1, 4096)
        concat = torch.cat([x, cnn_out], 1)
#         print('concat',concat.size())

        x = self.fc1(concat)
#         print('x2',x.size())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x,concat



class Resent18(nn.Module):
    def __init__(self, classes=450):
        super(Resent18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)
        self.model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)

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

    
def get_network_fn(name):
    networks_zoo = {
    'gcn': GCN(channel=4),
    'Resent18': Resent18(),
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