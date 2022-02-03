from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
from torch import nn
from efficientnet_pytorch import EfficientNet


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
    def __init__(self, inchannel=2, bits=1024):
        super(Resent18hashing, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, bits)
        self.model.conv1 = nn.Conv2d(inchannel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc.weight.data.normal_(0, 0.01)
        self.model.fc.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.model(x)
        return x

class Resent182Path(nn.Module):
    def __init__(self, inchannel1=2, inchannel2=2, bits=1024):
        super(Resent182Path, self).__init__()
        self.model1 = Resent18hashing(inchannel=inchannel1, bits=bits)
        self.model2 = Resent18hashing(inchannel=inchannel2, bits=bits)

    def forward(self, x, y):
        x = self.model1(x)
        y = self.model2(y)
        return x, y
class mobilenet_v3_largehashing(nn.Module):
    def __init__(self, inchannel=2,bits = 1024):
        super(mobilenet_v3_largehashing, self).__init__()
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier[0].in_features, bits)
        self.model.classifier.weight.data.normal_(0, 0.01)
        self.model.classifier.bias.data.fill_(0.0)
        
    def forward(self, x):
        x = self.model(x)
        return x    

class mobilenet_v3_largehashing2Path(nn.Module):
    def __init__(self, inchannel1=2, inchannel2=2,bits = 1024):
        super(mobilenet_v3_largehashing2Path, self).__init__()
        self.model1 = mobilenet_v3_largehashing(inchannel=inchannel1,bits = bits)
        self.model2 = mobilenet_v3_largehashing(inchannel=inchannel2,bits = bits)

    def forward(self, x,y):
        x = self.model1(x)
        y = self.model2(y)
        return x,y    
    
    
class efficientnet_b7hashing(nn.Module):
    def __init__(self, inchannel=3,bits = 1024):
        super(efficientnet_b7hashing, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.model._fc = nn.Linear(self.model._fc.in_features, bits)
        self.model._fc.weight.data.normal_(0, 0.01)
        self.model._fc.bias.data.fill_(0.0)
        
    def forward(self, x):
        x = self.model(x)
        return x


class efficientnet_b72Path(nn.Module):
    def __init__(self, inchannel1=2, inchannel2=2,bits = 1024):
        super(efficientnet_b72Path, self).__init__()
        self.model1 = efficientnet_b7hashing(inchannel=inchannel1,bits = bits)
        self.model2 = efficientnet_b7hashing(inchannel=inchannel2,bits = bits)

    def forward(self, x,y):
        x = self.model1(x)
        y = self.model2(y)
        return x,y


class efficientnet_b72Path_share(nn.Module):
    def __init__(self, inchannel1=2, inchannel2=2, bits=1024):
        super(efficientnet_b72Path_share, self).__init__()
        self.model1 = efficientnet_b7hashing(inchannel=inchannel1, bits=bits)

    def forward(self, x, y):
        x = self.model1(x)
        y = self.model1(y)
        return x, y

class Resent182Path_share(nn.Module):
    def __init__(self, inchannel1=2, inchannel2=2, bits=1024):
        super(Resent182Path_share, self).__init__()
        self.model1 = Resent18hashing(inchannel=inchannel1, bits=bits)

    def forward(self, x, y):
        x = self.model1(x)
        y = self.model1(y)
        return x, y

class CNNF(nn.Module):
    def __init__(self,inchannel=1, bits=128, use_dropout=False):
        super(CNNF, self).__init__()
        self.bits = bits 
        self.dropout = use_dropout 
        self.drop_prob = 0.5
        self.stride = 1 
        self.spatial_dim = (128,128)

        self.stnmod = STNModule.SpatialTransformer(inchannel, self.spatial_dim, 3)
        
        self.conv1 = nn.Conv2d(inchannel, 16, kernel_size=3, stride=4, padding=0, bias=False)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
                
        self.fc1 = nn.Linear(128*14*14, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, bits)#128 is the 0EER

    def forward(self, x):
        rois, affine_grid = self.stnmod(x)
#         print(rois.size(),'**********') #torch.Size([256, 2, 128, 128]) **********
        out = F.relu(self.conv1(rois))
#         print('conv 1',out.size())
        out = F.max_pool2d(out,kernel_size = 2, stride=1, padding=0)
#         print('max_pool 2',out.size())
        out = F.relu(self.conv2(out))
#         print('conv 2',out.size())
        out = F.max_pool2d(out,kernel_size = 2, stride=1, padding=0)
        out = F.relu(self.conv3(out))
#         print('conv 3',out.size())
        out = F.relu(self.conv4(out))
#         print('conv 4',out.size())
        out = F.relu(self.conv5(out))
#         print('conv 5',out.size())
        out = F.max_pool2d(out,kernel_size = 2, stride=1, padding=0)
        out = out.view(-1, 128*14*14)
        if self.dropout:
            out = F.dropout(self.fc1(out), p=0.5)
            out = F.dropout(self.fc2(out), p=0.5)
        else:
            out = self.fc1(out)
            out = self.fc2(out)
        out = self.fc3(out)
        return out

class CNNF2Path(nn.Module):
    def __init__(self, inchannel1=2, inchannel2=2, bits=128):
        super(CNNF2Path, self).__init__()
        self.model1 = CNNF(inchannel1,bits=bits)
        self.model2 = CNNF(inchannel2,bits=bits)

    def forward(self, x,y):
        x = self.model1(x)
        y = self.model2(y)
        return x,y
    