import numpy as np
import os
# import cv2
from matplotlib import pyplot as plt

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
import math
# read image
import PIL
from PIL import Image, ImageOps

ms_polyu_path = 'dataset/MS_PolyU/'
casia_path = 'dataset/CASIA-Multi-Spectral-PalmprintV1/images/'

r_img_path = ms_polyu_path + 'Red_ind/'
b_img_path = ms_polyu_path + 'Blue_ind/'
n_img_path = ms_polyu_path + 'NIR_ind/'
g_img_path = ms_polyu_path + 'Green_ind/'

tjp_path = 'dataset/Tongji_jpg/'
tjv_path = 'dataset/Palmvein_ROI_gray_128x128/session1/'
tjv_path2 = 'dataset/Palmvein_ROI_gray_128x128/session2/'

iitd_path = 'dataset/IITD/'

casia_path = 'dataset/Casia_M/'
casia_r_img_path = casia_path + 'casia_700/'
casia_b_img_path = casia_path + 'casia_460/'
casia_n_img_path = casia_path + 'casia_850/'


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
def part_init_polyu(istrain=True, train_ratio=1, sample_ratio=0.666, open_set_2nd=False):
    r_list = []
    b_list = []
    vein_list = []
    prints_list = []
    labels = []

    # split all data into train, test data
    #     train_ratio = 1
    #     sample_ratio = 0.333
    train_num = math.ceil(500 * train_ratio)
    sample_num = math.ceil(12 * sample_ratio)
    print("split train users:", train_num)
    print("split train samples:", sample_num, 'total sample:', 12)

#     users_permu = np.random.RandomState(seed=42).permutation(500)
    users_permu = np.array(range(500))
    users_permu_train = users_permu[:train_num]
    users_permu_test = users_permu[train_num:]
    print("users_permu_train:", users_permu_train)
    print("users_permu_test:", users_permu_test)

    sample_permu = np.random.RandomState(seed=42).permutation(12)
    sample_permu_train = sample_permu[:sample_num]
    sample_permu_test = sample_permu[sample_num:]
    print("sample_permu_train:", sample_permu_train)
    print("sample_permu_test:", sample_permu_test)

    if open_set_2nd:  # if openset settings, we train the incremental model on test, test should combine both 1st stage DS and 2nd stage DS
        users_permu_test = users_permu_train

    if istrain:
        for i in users_permu_train:
            for j in sample_permu_train:
                r_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(r_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))

                g_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(g_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))

                b_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(b_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))

                n_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(n_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))

                # rb = r_normed - b_normed * 0.5
                # rb =  (rb * 128+128).astype(np.uint8)

                imgprint = np.dstack((r_img, g_img, b_img))
                imgvein = np.dstack((n_img, n_img, n_img))

                vein_list.append(imgvein)
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, 500))
    #                 labels.append(i)
    else:
        for i in users_permu_train:
            for j in sample_permu_test:
                r_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(r_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))
                # r_normed = (r_img - r_img.min()) / (r_img.max()-r_img.min())

                g_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(g_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))

                b_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(b_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))
                # b_normed = (b_img - b_img.min()) / (b_img.max()-b_img.min())

                n_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(n_img_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))

                imgprint = np.dstack((r_img, g_img, b_img))
                imgvein = np.dstack((n_img, n_img, n_img))

                vein_list.append(imgvein)
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, 500))
    #                 labels.append(i)

    # return np.array(r_list), np.array(b_list), np.array(n_list), np.array(labels),np.array(r_list_test), np.array(b_list_test), np.array(n_list_test), np.array(labels_test)
    return vein_list, prints_list, labels


# Tongji palmprint是20*600=12000张，
# TJV 20 * 600
def part_init_tjppv(istrain=True, train_ratio=1, sample_ratio=0.666, open_set_2nd=False):
    r_list = []
    b_list = []
    vein_list = []
    prints_list = []
    labels = []

    # split all data into train, test data
    train_num = math.ceil(600 * train_ratio)
    sample_num = math.ceil(20 * sample_ratio)
    print("split train users:", train_num)
    print("split train samples:", sample_num)

#     users_permu = np.random.RandomState(seed=42).permutation(600)
    users_permu = np.array(range(600))
    users_permu_train = users_permu[:train_num]
    users_permu_test = users_permu[train_num:]
    print("users_permu_train:", users_permu_train)
    print("users_permu_test:", users_permu_test)

    sample_permu = np.random.RandomState(seed=42).permutation(20)
    sample_permu_train = sample_permu[:sample_num]
    sample_permu_test = sample_permu[sample_num:]
    print("sample_permu_train:", sample_permu_train)
    print("sample_permu_test:", sample_permu_test)

    if open_set_2nd:  # if openset settings, we train the incremental model on test, test should combine both 1st stage DS and 2nd stage DS
        users_permu_test = users_permu_train

    if istrain:
        for i in users_permu_train:
            for j in sample_permu_train:
                r_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(tjp_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))  # +1
                prints_list.append(np.dstack((r_img, r_img, r_img)))

                if j < 10:
                    sid = (i) * 10 + j + 1
                    r_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(tjv_path, "%05d.bmp" % (sid)))))
                    vein_list.append(np.dstack((r_img, r_img, r_img)))
                else:
                    sid = (i) * 10 + j - 10 + 1
                    r_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(tjv_path2, "%05d.bmp" % (sid)))))
                    vein_list.append(np.dstack((r_img, r_img, r_img)))
                labels.append(one_hot_embedding(i, 600))
    #                 labels.append(i)
    else:
        for i in users_permu_train:
            for j in sample_permu_test:
                r_img = np.array(
                    ImageOps.autocontrast(Image.open(os.path.join(tjp_path, "%04d_" % (i + 1) + "%04d.jpg" % (j + 1)))))
                prints_list.append(np.dstack((r_img, r_img, r_img)))
                if j < 10:
                    sid = (i) * 10 + j + 1
                    r_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(tjv_path, "%05d.bmp" % (sid)))))
                    vein_list.append(np.dstack((r_img, r_img, r_img)))
                else:
                    sid = (i) * 10 + j - 10 + 1
                    r_img = np.array(ImageOps.autocontrast(Image.open(os.path.join(tjv_path2, "%05d.bmp" % (sid)))))
                    vein_list.append(np.dstack((r_img, r_img, r_img)))
                labels.append(one_hot_embedding(i, 600))

    # return np.array(r_list), np.array(b_list), np.array(n_list), np.array(labels),np.array(r_list_test), np.array(b_list_test), np.array(n_list_test), np.array(labels_test)
    return vein_list, prints_list, labels


# IITD datasets是460*5=2300张。
def part_init_iitd(istrain=True, train_ratio=1, sample_ratio=0.666, open_set_2nd=False):
    r_list = []
    b_list = []
    vein_list = []
    prints_list = []
    labels = []

    # split all data into train, test data
    train_num = math.ceil(460 * train_ratio)
    sample_num = math.ceil(5 * sample_ratio)
    print("split train users:", train_num)
    print("split samples:", sample_num)

    # users_permu = np.random.RandomState(seed=42).permutation(460)
    users_permu = np.array(range(460))
    users_permu_train = users_permu[:train_num]
    users_permu_test = users_permu[train_num:]
    print("users_permu_train:", users_permu_train)
    print("users_permu_test:", users_permu_test)

    sample_permu = np.random.RandomState(seed=42).permutation(5)
    sample_permu_train = sample_permu[:sample_num]
    sample_permu_test = sample_permu[sample_num:]
    print("sample_permu_train:", sample_permu_train)
    print("sample_permu_test:", sample_permu_test)

    if open_set_2nd:  # if openset settings, we train the incremental model on test, test should combine both 1st stage DS and 2nd stage DS
        users_permu_test = users_permu_train

    if sample_num < 2:
        print('attention, testing sample not enough!')
    if istrain:
        for i in users_permu_train:
            fileid = (i) % 230 + 1
            for j in sample_permu_train:
                r_img = np.array(
                    Image.open(os.path.join(iitd_path, "%04d" % (i + 1), "%03d_" % fileid + "%01d.bmp" % (j + 1))))
                prints_list.append(np.dstack((r_img, r_img, r_img)))
                labels.append(one_hot_embedding(i, 460))
    #                 labels.append(i)
    else:
        for i in users_permu_train:
            fileid = (i) % 230 + 1
            for j in sample_permu_test:
                r_img = np.array(
                    Image.open(os.path.join(iitd_path, "%04d" % (i + 1), "%03d_" % fileid + "%01d.bmp" % (j + 1))))
                prints_list.append(np.dstack((r_img, r_img, r_img)))
                labels.append(one_hot_embedding(i, 460))

    # return np.array(r_list), np.array(b_list), np.array(n_list), np.array(labels),np.array(r_list_test), np.array(b_list_test), np.array(n_list_test), np.array(labels_test)
    return vein_list, prints_list, labels


def part_init_casiam(istrain=True, train_ratio=1, sample_ratio=0.666, open_set_2nd=False):
    r_list = []
    b_list = []
    vein_list = []
    prints_list = []
    labels = []

    # split all data into train, test data
    train_num = math.ceil(200 * train_ratio)  # 200 users
    sample_num = math.ceil(6 * sample_ratio)  # 200 users
    print("split train users:", train_num)
    print("split samples:", sample_num)

#     users_permu = np.random.RandomState(seed=42).permutation(200)
    users_permu = np.array(range(200))
    users_permu_train = users_permu[:train_num]
    users_permu_test = users_permu[train_num:]
    print("users_permu_train:", users_permu_train)
    print("users_permu_test:", users_permu_test)

    sample_permu = np.random.RandomState(seed=42).permutation(6)
    sample_permu_train = sample_permu[:sample_num]
    sample_permu_test = sample_permu[sample_num:]
    print("sample_permu_train:", sample_permu_train)
    print("sample_permu_test:", sample_permu_test)

    if open_set_2nd:  # if openset settings, we train the incremental model on test, test should combine both 1st stage DS and 2nd stage DS
        users_permu_test = users_permu_train

    if sample_num < 2:
        print('attention, testing sample not enough!')
    if istrain:
        for i in users_permu_train:
            for j in sample_permu_train:
                # XXX_(L/R) _ YYY_ZZ .jpg
                r_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(casia_r_img_path, "%03d_" % (i + 1) + "%02d.jpg" % (j + 1)))))

                b_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(casia_b_img_path, "%03d_" % (i + 1) + "%02d.jpg" % (j + 1)))))

                n_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(casia_n_img_path, "%03d_" % (i + 1) + "%02d.jpg" % (j + 1)))))

                imgprint = np.dstack((r_img, b_img, b_img))
                imgvein = np.dstack((n_img, n_img, n_img))

                vein_list.append(imgvein)
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, 200))
    #                 labels.append(i)
    else:
        for i in users_permu_train:
            for j in sample_permu_test:
                r_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(casia_r_img_path, "%03d_" % (i + 1) + "%02d.jpg" % (j + 1)))))

                b_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(casia_b_img_path, "%03d_" % (i + 1) + "%02d.jpg" % (j + 1)))))

                n_img = np.array(ImageOps.autocontrast(
                    Image.open(os.path.join(casia_n_img_path, "%03d_" % (i + 1) + "%02d.jpg" % (j + 1)))))

                imgprint = np.dstack((r_img, b_img, b_img))
                imgvein = np.dstack((n_img, n_img, n_img))

                vein_list.append(imgvein)
                prints_list.append(imgprint)
                labels.append(one_hot_embedding(i, 200))
    #                 labels.append(i)

    # return np.array(r_list), np.array(b_list), np.array(n_list), np.array(labels),np.array(r_list_test), np.array(b_list_test), np.array(n_list_test), np.array(labels_test)
    return vein_list, prints_list, labels


# r_list, b_list, n_list, labels,r_list_test, b_list_test, n_list_test, labels_test = part_init()
class load_data(Dataset):
    """Loads the Data."""

    def __init__(self, ds='polyu', training=True, train_ratio=1, sample_ratio=0.666):

        self.training = training
        #         r_list, b_list, n_list, labels,r_list_test, b_list_test, n_list_test, labels_test = part_init()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(),
            transforms.RandomAffine(30),
            transforms.ToTensor(),
            # transforms.Resize((128, 128)),#
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]), ])

        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((128, 128)),#
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]), ])

        if self.training:
            print('\n...... Train files loading\n')
            if ds == 'polyu':
                self.vein_list, self.prints_list, self.labels = part_init_polyu(istrain=True, train_ratio=train_ratio,
                                                                                sample_ratio=sample_ratio)
            elif ds == 'tjppv':
                self.vein_list, self.prints_list, self.labels = part_init_tjppv(istrain=True, train_ratio=train_ratio,
                                                                                sample_ratio=sample_ratio)
            elif ds == 'iitd':
                self.vein_list, self.prints_list, self.labels = part_init_iitd(istrain=True, train_ratio=train_ratio,
                                                                               sample_ratio=sample_ratio)
            elif ds == 'casiam':
                self.vein_list, self.prints_list, self.labels = part_init_casiam(istrain=True, train_ratio=train_ratio,
                                                                                 sample_ratio=sample_ratio)
            else:
                print('wrong DS', ds)
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            if ds == 'polyu':
                self.vein_list, self.prints_list, self.labels = part_init_polyu(istrain=False, train_ratio=train_ratio,
                                                                                sample_ratio=sample_ratio)
            elif ds == 'tjppv':
                self.vein_list, self.prints_list, self.labels = part_init_tjppv(istrain=False, train_ratio=train_ratio,
                                                                                sample_ratio=sample_ratio)
            elif ds == 'iitd':
                self.vein_list, self.prints_list, self.labels = part_init_iitd(istrain=False, train_ratio=train_ratio,
                                                                               sample_ratio=sample_ratio)
            elif ds == 'casiam':
                self.vein_list, self.prints_list, self.labels = part_init_casiam(istrain=False, train_ratio=train_ratio,
                                                                                 sample_ratio=sample_ratio)
            else:
                print('wrong DS', ds)
            print('\nTest files loaded ......\n')

    def __len__(self):
        return len(self.prints_list) if len(self.prints_list) > len(self.vein_list) else len(self.vein_list)

    def __getitem__(self, idx):

        if self.training:
            prints_img = self.transform(self.prints_list[idx])
            vein_img = self.transform(self.vein_list[idx])
        else:
            prints_img = self.transform_test(self.prints_list[idx])
            vein_img = self.transform_test(self.vein_list[idx])

        label = self.labels[idx]

        return prints_img, vein_img, label


# r_list, b_list, n_list, labels,r_list_test, b_list_test, n_list_test, labels_test = part_init()
class load_data_single_channel(Dataset):
    """Loads the Data."""

    def __init__(self, ds='polyu', training=True, train_ratio=0.9, sample_ratio=0.666):

        self.training = training
        #         r_list, b_list, n_list, labels,r_list_test, b_list_test, n_list_test, labels_test = part_init()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(),
            transforms.RandomAffine(30),
            transforms.ToTensor(),
            # transforms.Resize((128, 128)),#
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225]),
            transforms.Normalize([0.5], [0.5]),

        ])
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((128, 128)),#
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225]),
            transforms.Normalize([0.5], [0.5]),
        ])
        if self.training:
            print('\n...... Train files loading\n')
            if ds == 'iitd':
                self.vein_list, self.prints_list, self.labels = part_init_iitd(istrain=True, train_ratio=train_ratio,
                                                                               sample_ratio=sample_ratio)
            else:
                print('wrong DS', ds)
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            if ds == 'iitd':
                self.vein_list, self.prints_list, self.labels = part_init_iitd(istrain=False, train_ratio=train_ratio,
                                                                               sample_ratio=sample_ratio)
            else:
                print('wrong DS', ds)
            print('\nTest files loaded ......\n')

    def __len__(self):
        return len(self.prints_list) if len(self.prints_list) > len(self.vein_list) else len(self.vein_list)

    def __getitem__(self, idx):

        if self.training:
            prints_img = self.transform(self.prints_list[idx])
        else:
            prints_img = self.transform_test(self.prints_list[idx])

        label = self.labels[idx]

        return prints_img, label
