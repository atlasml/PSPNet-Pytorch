import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data

num_classes = 19
ignore_label = 255
root = '/mnt/vana/nabaviss/datasets/CITYSCAPES'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def make_dataset(quality, mode):
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items

class CityScapes(data.Dataset):
    def __init__(self,mode = None,simul_transform=None, transform=None, target_transform=None,resize_transform = None):
        self.imgs = make_dataset(quality = 'fine',mode = mode)

        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))

        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform
        self.resize_transform = resize_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        
        imgs,gts = self.imgs[index]
        imgs,gts = Image.open(imgs),Image.open(gts)
        imgs = np.asarray(imgs,dtype = np.float32)
        gts = np.array(gts)
        gts_copy = gts.copy()

        for k, v in self.id_to_trainid.items():
            gts_copy[gts == k] = v
        gts = Image.fromarray(gts_copy.astype(np.uint8))   

        if self.transform is not None:
            imgs = self.transform(imgs)
 
        if self.target_transform is not None:
            gts = self.target_transform(gts)
        
        return imgs,gts

    def __len__(self):
        return len(self.imgs)
