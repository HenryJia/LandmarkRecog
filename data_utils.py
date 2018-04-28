from urllib import request, error
from time import time
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.misc import imread

#from multiprocessing.pool import ThreadPool

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
