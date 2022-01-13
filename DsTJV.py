import numpy as np
import os
# import cv2
from  matplotlib import pyplot as plt

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

# dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import DataParallel
import tqdm
from torchvision import datasets, transforms

# read image
import PIL
from PIL import Image, ImageOps 

tjv_path = 'dataset/Palmvein_ROI_gray_128x128/session1/'

################ DATASET CLASS
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
# one_hot_embedding(1, 10)
def part_init(istrain=True):
    r_list = []
    b_list = []
    vein_list = []
    prints_list = []
    labels = []
    
        # split all data into train, test data
    train_ratio = 1
    train_num = int(600 * train_ratio)
    print("split train users:",train_num)
    if istrain:
        for i in tqdm.tqdm(range(train_num)):
            for j in range(7):
                sid = (i) * 10 + j + 1
                r_img = Image.open(os.path.join(tjv_path, "%05d.bmp"%(sid)))
                r_img =  np.array(ImageOps.autocontrast(r_img))
                imgprint = np.dstack((r_img,r_img,r_img))
                
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, train_num))
#                 labels.append(i)
    else:
        for i in tqdm.tqdm(range(train_num)):
            for j in range(7,10):
                sid = (i) * 10 + j
                r_img = Image.open(os.path.join(tjv_path, "%05d.bmp"%(sid)))
                r_img =  np.array(ImageOps.autocontrast(r_img))
                imgprint = np.dstack((r_img,r_img,r_img))
                
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, train_num))
#                 labels.append(i)



    # return np.array(r_list), np.array(b_list), np.array(n_list), np.array(labels),np.array(r_list_test), np.array(b_list_test), np.array(n_list_test), np.array(labels_test)
    return  vein_list,prints_list, labels

# r_list, b_list, n_list, labels,r_list_test, b_list_test, n_list_test, labels_test = part_init()
class load_data(Dataset):
    """Loads the Data."""
    def __init__(self, training=True):

        self.training = training
#         r_list, b_list, n_list, labels,r_list_test, b_list_test, n_list_test, labels_test = part_init()
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(),
        transforms.RandomAffine(30),
        transforms.ToTensor(),
#         transforms.Resize((224, 224)),# if resnet
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225]),
    ])
        self.transform_test = transforms.Compose([
        transforms.ToPILImage(),
#         transforms.Resize((224, 224)),# if resnet
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225]),
    ])
        if self.training:
            print('\n...... Train files loading\n')
            self.vein_list,self.prints_list, self.labels= part_init(istrain=True)
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            self.vein_list,self.prints_list, self.labels = part_init(istrain=False)
            print('\nTest files loaded ......\n')

    def __len__(self):
        return len(self.prints_list)

         
    def __getitem__(self, idx):

        if self.training:
            prints_img = self.transform(self.prints_list[idx])
        else:
            prints_img = self.transform_test(self.prints_list[idx])
        
        label = self.labels[idx]
        
        
        return prints_img,label
