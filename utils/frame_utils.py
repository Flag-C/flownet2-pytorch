import numpy as np
from os.path import *
from scipy.misc import imread
import flow_utils 
from tools import readPFM

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if len(im.shape)<3:
            im = im[:,:,np.newaxis]
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        im = readPFM(file_name)
        if len(im.shape)>=3:
            return im[:,:,:2]
        else:
            return im[:,:,np.newaxis]
    return []
