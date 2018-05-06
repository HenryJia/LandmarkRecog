from urllib import request, error
from time import time
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.misc import imread
from scipy.ndimage import zoom
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

class CSVDataset(Dataset):
    def __init__(self, dataframe, directory, submission = False):
        self.directory = directory
        self.dataframe = dataframe
        self.submission = submission

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
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        if self.submission:
            return img, idx
        return img, category

    def __len__(self):
        return len(self.dataframe)
