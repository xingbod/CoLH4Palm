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

ms_polyu_path = 'dataset/MS_PolyU/'
casia_path = 'dataset/CASIA-Multi-Spectral-PalmprintV1/images/'

r_img_path = ms_polyu_path + 'Red_ind/'
b_img_path =  ms_polyu_path + 'Blue_ind/'
n_img_path =  ms_polyu_path + 'NIR_ind/'
g_img_path =  ms_polyu_path + 'Green_ind/'

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
    train_num = int(500 * train_ratio)
    print("split train users:",train_num)
    if istrain:
        for i in tqdm.tqdm(range(train_num)):
            for j in range(8):
                r_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(r_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))
                r_normed = (r_img - r_img.min()) / (r_img.max()-r_img.min())
                
                g_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(g_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))

                b_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(b_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))
                b_normed = (b_img - b_img.min()) / (b_img.max()-b_img.min())

                n_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(n_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))
                
                rb = r_normed - b_normed * 0.5
                rb =  (rb * 128+128).astype(np.uint8)

                imgprint = np.dstack((r_img,g_img,b_img))
                imgvein = np.dstack((rb, n_img))
                
                vein_list.append(imgvein)
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, train_num))
#                 labels.append(i)
    else:
        for i in tqdm.tqdm(range(train_num)):
            for j in range(8,12):
                r_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(r_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))
                r_normed = (r_img - r_img.min()) / (r_img.max()-r_img.min())
                
                g_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(g_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))

                b_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(b_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))
                b_normed = (b_img - b_img.min()) / (b_img.max()-b_img.min())

                n_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(n_img_path, "%04d_"%(i+1)+"%04d.jpg"%(j+1)))))
                
                rb = r_normed - b_normed * 0.5
                rb =  (rb * 128+128).astype(np.uint8)
                imgprint = np.dstack((r_img,g_img,b_img))
                imgvein = np.dstack((rb, n_img))
                
                vein_list.append(imgvein)
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
        return len(self.vein_list)

         
    def __getitem__(self, idx):

        if self.training:
            prints_img = self.transform(self.prints_list[idx])
            vein_img = self.transform(self.vein_list[idx])
        else:
            prints_img = self.transform_test(self.prints_list[idx])
            vein_img = self.transform_test(self.vein_list[idx])
        
        label = self.labels[idx]
        
        n_img = np.dstack((prints_img[0,:,:],prints_img[1,:,:],prints_img[2,:,:],vein_img[0,:,:],vein_img[1,:,:]))
        
        return n_img,label
