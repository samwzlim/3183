"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)

class CustomReferenceDataset(data.Dataset):
    def __init__(self, root, mask_dir, img_size, transform=None):
        self.samples, self.masks, self.targets = self._make_dataset(root, mask_dir)
        self.transform = transform
        self.img_size = img_size

    def _make_dataset(self, root, root_mask_dir):
        domains = os.listdir(root)
        fnames, fnames2, masks, masks2, labels = [], [], [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            mask_dir = os.path.join(root_mask_dir, domain)
            cls_fnames = listdir(class_dir)
            mask_fnames = listdir(mask_dir)
            cls_fnames.sort()
            mask_fnames.sort()

            fnames += cls_fnames
            masks+= mask_fnames

            fnames2 = fnames
            masks2 = masks
            c = list(zip(fnames2, masks2))
            random.shuffle(c)
            fnames2, masks2 = zip(*c)
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), list(zip(masks,masks2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        mask, mask2 = self.masks[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        mask = Image.open(mask)
        mask2 = Image.open(mask2)
        img,mask = random_transform(img,mask)
        img2,mask2 = random_transform(img2,mask2)
        return img, img2, mask, mask2, label

    def __len__(self):
        return len(self.targets)
    
class CustomDataset(data.Dataset):
    def __init__(self, root, mask_dir, img_size, transform_img=None, transform_mask=None, test=False):
        self.samples, self.masks, self.targets = self._make_dataset(root, mask_dir)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.test = test
        self.img_size = img_size

    def _make_dataset(self, root, root_mask_dir):
      domains = os.listdir(root)
      fnames, masks, labels = [], [], []
      for idx, domain in enumerate(sorted(domains)):
          class_dir = os.path.join(root, domain)
          mask_dir = os.path.join(root_mask_dir, domain)
          cls_fnames = listdir(class_dir)
          mask_fnames = listdir(mask_dir)
          # Ensure that cls_fnames and mask_fnames are sorted in the same order
          cls_fnames.sort()
          mask_fnames.sort()

          fnames += cls_fnames
          masks+= mask_fnames
          labels += [idx] * len(cls_fnames)
      return list(fnames), list(masks), labels


    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        mask_fname = self.masks[index]
        img = Image.open(fname).convert('RGB')
        mask = Image.open(mask_fname)
        if not self.test:
            img, mask = random_transform(img,mask,self.img_size)
        if self.test and (self.transform_img,self.transform_mask) != (None,None):
            img = self.transform_img(img)
            mask = self.transform_mask(mask)
        return img, mask, label

    def __len__(self):
        return len(self.targets)
    
class CustomEvalDataset(data.Dataset):
    def __init__(self, root, mask_dir, img_size, transform_img=None, transform_mask=None):
        self.samples, self.masks= self._make_dataset(root, mask_dir)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.img_size = img_size

    def _make_dataset(self, root, root_mask_dir):
      fnames, masks = [], []
      cls_fnames = listdir(root)
      mask_fnames = listdir(root_mask_dir)
      # Ensure that cls_fnames and mask_fnames are sorted in the same order
      cls_fnames.sort()
      mask_fnames.sort()
      fnames += cls_fnames
      masks+= mask_fnames
      return list(fnames), list(masks)


    def __getitem__(self, index):
        fname = self.samples[index]
        mask_fname = self.masks[index]
        img = Image.open(fname).convert('RGB')
        mask = Image.open(mask_fname)
        if (self.transform_img,self.transform_mask) != (None,None):
            img = self.transform_img(img)
            mask = self.transform_mask(mask)
        return img, mask

    def __len__(self):
        return len(self.samples)

def random_transform(A,A_mask, img_size=256, prob=0.5):
    # Random resized crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        A, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    
    if (random.random() < prob):
        A = TF.crop(A, i, j, h, w)
        A_mask = TF.crop(A_mask, i, j, h, w)
    

    # resize
    A = TF.resize(A, [img_size,img_size], antialias=True)
    A_mask = TF.resize(A_mask, [img_size,img_size],antialias=True)

    # Random horizontal flipping
    if random.random() > 0.5:
        A = TF.hflip(A)
        A_mask = TF.hflip(A_mask)

    # to tensor
    A = TF.to_tensor(A)
    A_mask = TF.to_tensor(A_mask)

    # normalize only images
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    return A, A_mask

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(args, root, mask_dir, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size],antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = CustomDataset(root, mask_dir, img_size, transform)
    elif which == 'reference':
        dataset = CustomReferenceDataset(root, mask_dir, img_size, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_fid_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
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
        transforms.Resize([img_size, img_size],antialias=True),
        transforms.Resize([height, width],antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def get_eval_loader(root, mask_dir, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform_img = transforms.Compose([
        transforms.Resize([img_size, img_size],antialias=True),
        transforms.Resize([height, width],antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_mask = transforms.Compose([
        transforms.Resize([img_size, img_size],antialias=True),
        transforms.Resize([height, width],antialias=True),
        transforms.ToTensor(),
    ])

    dataset = CustomEvalDataset(root, mask_dir, img_size=img_size, transform_img=transform_img, transform_mask=transform_mask)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(args, root, mask_dir, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform_img = transforms.Compose([
        transforms.Resize([img_size, img_size],antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_mask = transforms.Compose([
        transforms.Resize([img_size, img_size],antialias=True),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(root, mask_dir, img_size, transform_img, transform_mask, test=True)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, args, loader, loader_ref=None, latent_dim=16, mode=''):
        self.args = args
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, mask, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, mask, y = next(self.iter)
        return x, mask, y

    def _fetch_refs(self):
        try:
            x, x2, x_mask, x_mask2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, x_mask, x_mask2, y = next(self.iter_ref)
        return x, x2, x_mask, x_mask2, y 
       

    def __next__(self):
        x, mask, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, x_ref_mask, x_ref2_mask, y_ref = self._fetch_refs()

            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)

            inputs = Munch(x_src=x, x_mask=mask, y_src=y, y_ref=y_ref,
                        x_ref=x_ref, x_ref_mask=x_ref_mask, x_ref2=x_ref2,
                        x_ref2_mask=x_ref2_mask,z_trg=z_trg, z_trg2=z_trg2)
                
        elif self.mode == 'val':
            x_ref, x_ref_mask, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, x_mask=mask, y_src=y,
                           x_ref=x_ref, x_ref_mask=x_ref_mask, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, mask=mask, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})