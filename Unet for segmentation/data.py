import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

#Global parameters
H = 256
W = 256

#Creating Directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    split = 0.2
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=35)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=35)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=35)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=35)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "dataset/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")




