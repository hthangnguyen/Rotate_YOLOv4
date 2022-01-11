# Reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/datasets.py

import glob
import random
import os
import numpy as np
import cv2 as cv
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from tools.plot import xywha2xyxyxyxy
from tools.augments import vertical_flip, horisontal_flip, rotate, gaussian_noise, hsv


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageDataset(Dataset):
    def __init__(self, folder_path, img_size=416):
#         self.files = sorted(glob.glob("%s/*.png" % folder_path))
        self.files = sorted(glob.glob("%s/*.jpg" % folder_path))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #  Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #  Width in {320, 416, 512, 608, ... 320 + 96 * m}
        img_path = self.files[index % len(self.files)]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        #transform = transforms.ToPILImage(mode="RGB")
        #image = transform(img)
        #image.show()
        return img_path, img


class ListDataset(Dataset):
    def __init__(self, list_path, labels, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        self.img_files = list_path

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.labels = labels
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution
        if self.augment:
            if np.random.random() < 0.25:
                img = gaussian_noise(img, 0.0, np.random.random())
            if np.random.random() < 0.25:
                img = hsv(img)
        img, pad = pad_to_square(img, 0)

        # show image
        # transform = transforms.ToPILImage(mode="RGB")
        # image = transform(img)
        # image.show()

        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
#             label = torch.from_numpy(np.array(self.labels[index]))

#             x1, y1, x2, y2, x3, y3, x4, y4 = \
#                 boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6], boxes[:, 7]
            label, x, y, w, h, theta = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
            num_targets = len(boxes)

#             print(label, x, y, w, h, theta)
            targets = torch.zeros((len(boxes), 7))
            targets[:, 1] = label
            targets[:, 2] = x
            targets[:, 3] = y
            targets[:, 4] = w
            targets[:, 5] = h
            targets[:, 6] = theta
        else:
            targets = torch.zeros((1, 7))
            targets[:, 1] = -1
            return img_path, img, targets

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = rotate(img, targets)
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            if np.random.random() < 0.5:
                img, targets = vertical_flip(img, targets)
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


def split_data(data_dir, img_size, batch_size=4, shuffle=True, augment=True, multiscale=True):
    dataset = ImageFolder(data_dir)

    classes = [[] for _ in range(len(dataset.classes))]

    for x, y in dataset.samples:
        classes[int(y)].append(x)

    inputs, labels = [], []

    for i, data in enumerate(classes):  # 讀取每個類別中所有的檔名 (i: label, data: filename)

        for x in data:
            inputs.append(x)
            labels.append(i)

    dataset = ListDataset(inputs, labels, img_size=img_size, augment=augment, multiscale=multiscale)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   pin_memory=True, collate_fn=dataset.collate_fn)

    return dataset, dataloader