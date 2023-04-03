import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import imageio

"""Global Parameters"""
H = 512
W = 512

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (h, w)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    return x

class ImageDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = read_image(self.X[index])
        y = read_mask(self.Y[index])
        return x, y

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    split = 0.2

    images = sorted(glob(os.path.join(path, "images", "*.png")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=35)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=35)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=35)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=35)
    return (train_x, train_y), (valid_x, valid_y),  (test_x, test_y)

