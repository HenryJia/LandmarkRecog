from urllib import request, error
from time import time
import random

import numpy as np
from PIL import Image
from io import BytesIO
from scipy.misc import imread
from scipy.ndimage.interpolation import zoom, rotate
import cv2

import torch
from torch.utils.data import Dataset

from tqdm import tqdm


def grab(url):
    response = request.urlopen(url)
    image_data = response.read()

    img = Image.open(BytesIO(image_data))

    img = img.convert('RGB')

    return np.array(img)


def sample(dataframe):
    img = None
    while(img is None):
        try:
            img = grab(dataframe.sample(1).iloc[0, 1])
        except:
            img = None

    return img


def read_img(img_id):
    try: # Try reading the downloaded data first
        return imread('./test/' + img_id + '.jpg')
    except: # Otherwise, try grabbing it off the web
        try:
            return grab(test_dict[img_id])
        except: # OK, so the image doesn't exist, just return a None
            return None


def random_crop(img, size):
    idx = np.random.randint(0, img.shape[0] - size[0])
    idy = np.random.randint(0, img.shape[1] - size[1])

    return img[idx:idx + size[0], idy:idy + size[1]]


class RandomFlip(object):
    """Horizontally flip the given NumPy array randomly with a given probability.
    Args:
        axis (positive integer): axis to flip along
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, axis = 0, p = 0.5):
        self.axis = axis
        self.p = p

    def __call__(self, x):
        """
        Args:
            x (NumPy array): array to be flipped.
        Returns:
            NumPy array: Randomly flipped NumPy array.
        """
        if random.random() < self.p:
            return np.flip(x, axis = self.axis).copy()
        return x


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (min, max): Range of degrees to select from.
    """

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = random.uniform(self.degrees[0], self.degrees[1])

        return rotate(img, angle, order = 1, reshape = False)


class CSVDataset(Dataset): # Note: All torchvision transforms are just classes with a __call__ attribute anyway, so they can be used here
    def __init__(self, dataframe, directory, transforms = None, submission = False):
        self.directory = directory
        self.dataframe = dataframe
        self.submission = submission
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        url = row['url']
        idx = row['id']
        if self.submission:
            category = torch.LongTensor([-1])[0]
        else:
            category = torch.LongTensor([row['landmark_id']])[0]

        img = imread(self.directory + idx + '.jpg')
        #img = zoom(img, (224.0 / img.shape[0], 224.0 / img.shape[1], 1), order = 1)
        #img = Image.open(self.directory + idx + '.jpg')

        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
        if self.transforms is not None:
            img = self.transforms(img)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        if self.submission:
            return img, idx
        return img, category

    def __len__(self):
        return len(self.dataframe)
