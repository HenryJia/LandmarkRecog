from urllib import request, error
from time import time
import numpy as np
from PIL import Image
from io import BytesIO

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
