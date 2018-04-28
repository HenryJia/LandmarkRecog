import pickle

import numpy as np
import cv2

def save_orb(fn, orb):
    kp, desc = orb

    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)

    with open(fn + '.kp', 'wb') as f:
        pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

    np.save(fn, desc)


def load_orb(fn):
    with open(fn + '.kp', 'rb') as f:
        index = pickle.load(f)

    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5]) 
        kp.append(temp)

    desc = np.load(fn + '.npy')

    return (kp, desc)
