import os
import random
import cv2  # For image processing
from pathlib import Path
from itertools import chain
from munch import Munch
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import albumentations as A  # Albumentations for advanced data augmentation
from albumentations.pytorch import ToTensorV2

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

def histogram_equalization(img):
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel, cr, cb = cv2.split(img_y_cr_cb)
    y_channel_eq = cv2.equalizeHist(y_channel)
    img_y_cr_cb_eq = cv2.merge((y_channel_eq, cr, cb))
    img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
    return img_eq

class AlbumentationsDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = histogram_equalization(img)  # Apply histogram equalization
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img

    def __len__(self):
        return len(self.samples)

def get_train_loader(root, which='source', img_size=256, batch_size=8, prob=0.5, num_workers=4):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(),
        A.ColorJitter(),
        A.Cutout(num_holes=1, max_h_size=img_size // 8, max_w_size=img_size // 8),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    if which == 'source':
        dataset = AlbumentationsDataset(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    return data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)

def get_eval_loader(root, img_size=256, batch_size=32, imagenet_normalize=True, shuffle=True, num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)

def get_test_loader(root, img_size=256, batch_size=32, shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref, x_ref=x_ref, x_ref2=x_ref2, z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
