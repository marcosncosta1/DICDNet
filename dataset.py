import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import PIL
from PIL import Image

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2.0 - 1.0
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1]
        if vflip:
            img = img[::-1, :]
        return img

    return [_augment(a) for a in args]

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize, length, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask  # shape [512, 512, 90] presumably
        self.patch_size = patchSize
        self.sample_num = length
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        # Get the ground-truth file path from the txt file (e.g., "000304_03_01/151/82.h5")
        gt_dir = self.mat_files[idx % self.file_num].strip()

        # Build absolute path for the ground-truth file
        gt_absdir = os.path.join(self.dir, 'train_640geo', gt_dir)

        # Choose a random mask index (0-89) for the metal-corrupted file.
        random_mask = random.randint(0, 89)
        # Extract the folder from the ground-truth path (e.g., "000304_03_01/151")
        folder = os.path.dirname(gt_dir)
        # Construct the absolute path for the metal-corrupted file, e.g., "data/train/train_640geo/000304_03_01/151/7.h5"
        abs_dir = os.path.join(self.dir, 'train_640geo', folder, f"{random_mask}.h5")

        # Attempt to open the files; if an OSError occurs, skip to the next index.
        try:
            # Open the ground-truth file
            with h5py.File(gt_absdir, 'r') as gt_file:
                Xgt = gt_file['image'][()]
            # Open the metal-corrupted file
            with h5py.File(abs_dir, 'r') as f:
                Xma = f['ma_CT'][()]
                XLI = f['LI_CT'][()]
        except OSError:
            new_idx = (idx + 1) % self.file_num
            return self.__getitem__(new_idx)

        # Read the metal mask corresponding to the chosen random mask
        M512 = self.train_mask[:, :, random_mask]
        # Resize mask to 416 x 416 using bilinear interpolation
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))

        # Clip and normalize the CT images (values assumed in [0,1])
        Xgtclip = np.clip(Xgt, 0, 1)
        Xgtnorm = Xgtclip
        Xmaclip = np.clip(Xma, 0, 1)
        Xmanorm = Xmaclip
        XLIclip = np.clip(XLI, 0, 1)
        XLInorm = XLIclip

        # Convert images to a 0-255 scale
        O = Xmanorm * 255.0
        B = Xgtnorm * 255.0
        LI = XLInorm * 255.0

        # Randomly crop the image to patch_size
        O, row, col = self.crop(O)
        B = B[row: row + self.patch_size, col: col + self.patch_size]
        LI = LI[row: row + self.patch_size, col: col + self.patch_size]
        M = M[row: row + self.patch_size, col: col + self.patch_size]

        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)

        # Optionally perform augmentation (horizontal and vertical flips)
        O, B, LI, Mask = augment(O, B, LI, Mask)

        # Add a channel dimension and transpose to (C, H, W)
        O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
        B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
        LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))

        # Create non-metal region mask as 1 - Mask
        non_Mask = 1 - Mask

        return (
            torch.from_numpy(O.copy()),
            torch.from_numpy(B.copy()),
            torch.from_numpy(LI.copy()),
            torch.from_numpy(non_Mask.copy())
        )

    def crop(self, img):
        h, w = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if h == p_h and w == p_w:
            # no crop needed
            return img, 0, 0
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
            O = img[r: r + p_h, c: c + p_w]
            return O, r, c
