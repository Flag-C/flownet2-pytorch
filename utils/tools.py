# freda (todo) : 

import inspect
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from inspect import isclass
from os.path import *

import numpy as np
import torch
from pytz import timezone


def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

def module_to_dict(module, exclude=[]):
        return dict([(x, getattr(module, x)) for x in dir(module)
                     if isclass(getattr(module, x))
                     and x not in exclude
                     and getattr(module, x) not in exclude])

class TimerBlock: 
    def __init__(self, title):
        print("{}".format(title))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")


    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string))
    
    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n"%(string))
        fid.close()

def add_arguments_for_module(parser, module, argument_for_class, default, skip_params=[], parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_class.capitalize())

    module_dict = module_to_dict(module)
    argument_group.add_argument('--' + argument_for_class, type=str, default=default, choices=module_dict.keys())
    
    args, unknown_args = parser.parse_known_args()
    class_obj = module_dict[vars(args)[argument_for_class]]

    argspec = inspect.getargspec(class_obj.__init__)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_class, arg)
        if arg not in skip_params + ['self', 'args']:
            if arg in parameter_defaults.keys():
                    argument_group.add_argument('--{}'.format(cmd_arg), type=type(parameter_defaults[arg]), default=parameter_defaults[arg])
            elif (defaults is not None and i < len(defaults)):
                if not arg in ['lr','momentum']:
                    argument_group.add_argument('--{}'.format(cmd_arg), type=type(defaults[i]), default=defaults[i])
                else:
                    argument_group.add_argument('--{}'.format(cmd_arg), type=float, default=defaults[i])
            else:
                print("[Warning]: non-default argument '{}' detected on class '{}'. This argument cannot be modified via the command line"
                        .format(arg, module.__class__.__name__))
            # We don't have a good way of dealing with inferring the type of the argument
            # TODO: try creating a custom action and using ast's infer type?
            # else:
            #     argument_group.add_argument('--{}'.format(cmd_arg), required=True)

def kwargs_from_args(args, argument_for_class):
    argument_for_class = argument_for_class + '_'
    return {key[len(argument_for_class):]: value for key, value in vars(args).items() if argument_for_class in key and key != argument_for_class + 'class'}

def format_dictionary_of_losses(labels, values):
    try:
        string = ', '.join([('{}: {:' + ('.3f' if value >= 0.001 else '.1e') +'}').format(name, value) for name, value in zip(labels, values)])
    except (TypeError, ValueError) as e:
        print(zip(labels, values))
        string = '[Log Error] ' + str(e)

    return string


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.next()
        self.last_duration = (time.time() - start)
        return n

    next = __next__

def gpumemusage():
    gpu_mem = subprocess.check_output("nvidia-smi | grep MiB | cut -f 3 -d '|'", shell=True).replace(' ', '').replace('\n', '').replace('i', '')
    all_stat = [float(a) for a in gpu_mem.replace('/','').split('MB')[:-1]]

    gpu_mem = ''
    for i in range(len(all_stat)/2):
        curr, tot = all_stat[2*i], all_stat[2*i+1]
        util = "%1.2f"%(100*curr/tot)+'%'
        cmem = str(int(math.ceil(curr/1024.)))+'GB'
        gmem = str(int(math.ceil(tot/1024.)))+'GB'
        gpu_mem += util + '--' + join(cmem, gmem) + ' '
    return gpu_mem


def update_hyperparameter_schedule(args, epoch, global_iteration, optimizer):
    if args.schedule_lr_frequency > 0:
        for param_group in optimizer.param_groups:
            if (epoch + 1) % args.schedule_lr_frequency == 0:
                param_group['lr'] /= float(args.schedule_lr_fraction)
                param_group['lr'] = float(np.maximum(param_group['lr'], 0.000001))

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

UNKNOWN_FLOW_THRESH = 1e7

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map(w*h*2)
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(file_name):
    file = open(file_name)
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()
    return np.reshape(data, shape)


'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file_name, image, scale=1):
    file = open(file_name)
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)
