from __future__ import division

import torch
from torchvision import transforms
import numpy as np
import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
import random

def batch_random_mask(images_tensor, patch_size=16, mask_prob=0.2):
    B, C, H, W = images_tensor.shape
    mask_one_hot = []
    for i in range(B):
        image_tensor , one_hot= random_mask(images_tensor[i], patch_size, mask_prob)
        images_tensor[i, :, :, :] = image_tensor[:, :, :]
        mask_one_hot.append(one_hot)
    return images_tensor,mask_one_hot

def random_mask(image_tensor, patch_size=16, mask_prob=0.75):
    _, h, w = image_tensor.size()
    mask = torch.ones_like(image_tensor)

    # 计算需要遮挡的patch数量
    total_patches = (h // patch_size) * (w // patch_size)
    num_patches_to_mask = int(total_patches * mask_prob)

    # 生成所有可能的patch的坐标
    patch_coords = [(i, j) for i in range(0, h, patch_size) for j in range(0, w, patch_size)]

    # 随机选择并遮挡patch
    selected_coords = random.sample(patch_coords, num_patches_to_mask)
    for coord in selected_coords:
        top, left = coord
        mask[:, top:top + patch_size, left:left + patch_size] = 0

    # 获取被遮挡的patch的序号
    masked_patch_indices = [patch_coords.index(coord) for coord in selected_coords]
    masked_one_hot = np.zeros((14*14))
    masked_one_hot[masked_patch_indices] = 1
    return image_tensor * mask ,masked_one_hot

def data_transform(name, size=224):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []

    if 'resize_random_crop' in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),

            # ###
        ])
    elif 'resize_center_crop' in name:
        transform.extend(
            transforms.Resize(size),
            transforms.CenterCrop(size),
        )
    elif 'resize_only' in name:
        transform.extend([
            transforms.Resize((size, size)),
        ])
    elif 'resize' in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
        ])
    else:
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])

    if 'colorjitter' in name:
        transform.extend(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2)
        )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform

def patchCrop(img, size, lam):
    n,c,w,h = img.shape

    cut_rate = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rate)
    cut_h = np.int(h * cut_rate)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    patch = img[:,:,bbx1:bbx2,bby1:bby2]
    torch_resize = transforms.Resize(int(size * 8. / 7.))
    patch = torch_resize(patch)
    return patch