import random
from glob import glob
from os.path import *

import numpy as np
import torch
import torch.utils.data as data

import utils.frame_utils as frame_utils


class StaticRandomCrop(object):
    def __init__(self, size):
        self.th, self.tw = size
        self.h1 = None
        self.w1 = None
    def __call__(self, img):
        h, w, _ = img.shape
        if self.h1 is None:
            self.h1 = random.randint(0, h - self.th)
        if self.w1 is None:
            self.w1 = random.randint(0, w - self.tw)
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, size):
        self.th, self.tw = size
    def __call__(self, img):
        h, w, _ = img.shape
        return img[(h-self.th)/2:(h+self.th)/2, (w-self.tw)/2:(w+self.tw)/2,:]

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            flow = cropper(flow)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)/2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    if self.is_cropped:
        cropper = StaticRandomCrop(self.crop_size)
        images = map(cropper, images)
        flow = cropper(flow)
    else:
        cropper = StaticCenterCrop(self.render_size)
        images = map(cropper, images)
        flow = cropper(flow)

    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/data/qizhicai/flyingthings', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
    image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

    flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
    flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

    assert (len(image_dirs) == len(flow_dirs))

    self.image_list = []
    self.flow_list = []

    for idir, fdir in zip(image_dirs, flow_dirs):
        images = sorted( glob(join(idir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.pfm')) )
        for i in range(len(flows)-1):
            self.image_list += [ [ images[i], images[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)
    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    if self.is_cropped:
        cropper = StaticRandomCrop(self.crop_size)
        images = map(cropper, images)
        flow = cropper(flow)
    else:
        cropper = StaticCenterCrop(self.render_size)
        images = map(cropper, images)
        flow = cropper(flow)

    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class Driving(data.Dataset):
    def __init__(self, args, is_cropped, root='/data/qizhicai/driving', dstype='frames_cleanpass', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image_dirs = sorted(glob(join(root, dstype, '*/*/*')))
        image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

        flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/*/*/*')))
        flow_dirs = sorted(
            [join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

        assert (len(image_dirs) == len(flow_dirs))

        self.image_list = []
        self.flow_list = []

        for idir, fdir in zip(image_dirs, flow_dirs):
            images = sorted(glob(join(idir, '*.png')))
            flows = sorted(glob(join(fdir, '*.pfm')))
            for i in range(len(flows) - 1):
                self.image_list += [[images[i], images[i + 1]]]
                self.flow_list += [flows[i]]

        assert len(self.image_list) == len(self.flow_list)

        print(len(self.image_list))
        self.size = len(self.image_list)
        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
            self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) / 64) * 64
            self.render_size[1] = ((self.frame_size[1]) / 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            flow = cropper(flow)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class Monkaa(data.Dataset):
    def __init__(self, args, is_cropped, root='~/data/monkaa', dstype='frames_cleanpass', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image_dirs = sorted(glob(join(root, dstype, '*')))
        image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

        flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/*')))
        flow_dirs = sorted(
            [join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

        assert (len(image_dirs) == len(flow_dirs))

        self.image_list = []
        self.flow_list = []

        for idir, fdir in zip(image_dirs, flow_dirs):
            images = sorted(glob(join(idir, '*.png')))
            flows = sorted(glob(join(fdir, '*.pfm')))
            for i in range(len(flows) - 1):
                self.image_list += [[images[i], images[i + 1]]]
                self.flow_list += [flows[i]]

        assert len(self.image_list) == len(self.flow_list)

        print(len(self.image_list))
        self.size = len(self.image_list)
        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
            self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) / 64) * 64
            self.render_size[1] = ((self.frame_size[1]) / 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            flow = cropper(flow)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class KITTI(data.Dataset):
    def __init__(self, args, is_cropped=False, root='~/data/data_scene_flow', dstype='training', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image_dir = join(root, dstype,'image_2')

        flow_dir = join(root, dstype, 'flow_occ')

        self.image_list = []
        self.flow_list = []

        images_1 = sorted(glob(join(image_dir, '*_10.png')))
        images_2 = sorted(glob(join(image_dir, '*_11.png')))
        flows = sorted(glob(join(flow_dir, '*.png')))
        assert len(images_1)==len(images_2)
        assert len(flows)==len(images_1)

        for i in range(len(flows)):
            self.image_list += [[images_1[i], images_2[i]]]
            self.flow_list += [flows[i]]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)
        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
            self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) / 64) * 64
            self.render_size[1] = ((self.frame_size[1]) / 64) * 64

        args.inference_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index]).astype(float)

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            flow = cropper(flow)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)
        flow = flow[:2,:,:]
        flow = flow-128
        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates


class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/data/qizhicai/ChairsSDHom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.pfm') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = np.flipud(flow)
    flow = flow[::-1,:,:]
    images = [img1, img2]
    if self.is_cropped:
        cropper = StaticRandomCrop(self.crop_size)
        images = map(cropper, images)
        flow = cropper(flow)
    else:
        cropper = StaticCenterCrop(self.render_size)
        images = map(cropper, images)
        flow = cropper(flow)

    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class mixchairsandthings(data.Dataset):
    def __init__(self, args, is_cropped=False, root = ' ', replicates=1):
        self.datasets = (FlyingThings(args=args,is_cropped=is_cropped,replicates=replicates,root = '/data/qizhicai/flyingthings'),
                         ChairsSDHom(args=args,is_cropped=is_cropped,replicates=replicates,root = '/data/qizhicai/ChairsSDHom/data'))
    def __getitem__(self, i):
        frac = np.random.random()
        if frac>0.25:
            return self.datasets[1].__getitem__(np.random.randint(0,len(self.datasets[1])))
        else:
            return self.datasets[0].__getitem__(np.random.randint(0, len(self.datasets[0])))

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    for i in range(len(images)-1):
        im1 = images[i]
        im2 = images[i+1]
        self.image_list += [ [ im1, im2 ] ]

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    images = [img1, img2]
    if self.is_cropped:
        cropper = StaticRandomCrop(self.crop_size)
        images = map(cropper, images)
    else:
        cropper = StaticCenterCrop(self.render_size)
        images = map(cropper, images)

    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates
