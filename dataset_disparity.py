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

class FlyingThings(data.Dataset):
    def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
        left_image_dirs = sorted([join(f, 'left') for f in image_dirs])
        right_image_dirs = sorted([join(f, 'right') for f in image_dirs])

        disparity_dirs = sorted(glob(join(root, 'disparity/TRAIN/*/*')))
        disparity_dirs = sorted([join(f,'left') for f in disparity_dirs])

        assert (len(left_image_dirs) == len(right_image_dirs))
        assert (len(left_image_dirs) == len(disparity_dirs))

        self.image_list = []
        self.disparity_list = []

        for ldir, rdir, ddir in zip(left_image_dirs, right_image_dirs, disparity_dirs):
            l_images = sorted( glob(join(ldir, '*.png')))
            r_images = sorted( glob(join(rdir, '*.png')))
            disparities = sorted(glob(join(ddir, '*.pfm')))
            for i in range(len(disparities)):
                self.image_list += [ [ l_images[i], r_images[i] ] ]
                self.disparity_list += [disparities[i]]

        assert len(self.image_list) == len(self.disparity_list)

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

        disparity = frame_utils.read_gen(self.disparity_list[index])

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            disparity = cropper(disparity)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            disparity = cropper(disparity)

        images = np.array(images).transpose(3,0,1,2)
        disparity = disparity.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        disparity = torch.from_numpy(disparity.astype(np.float32))

        return images, disparity

    def __len__(self):
        return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class Driving(data.Dataset):
    def __init__(self, args, is_cropped, root='~/data/driving', dstype='frames_cleanpass', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        image_dirs = sorted(glob(join(root, dstype, '*/*/*')))
        left_image_dirs = sorted([join(f, 'left') for f in image_dirs])
        right_image_dirs = sorted([join(f, 'right') for f in image_dirs])

        disparity_dirs = sorted(glob(join(root, 'disparity/*/*/*')))
        disparity_dirs = sorted([join(f, 'left') for f in disparity_dirs])

        assert (len(left_image_dirs) == len(right_image_dirs))
        assert (len(left_image_dirs) == len(disparity_dirs))

        self.image_list = []
        self.disparity_list = []

        for ldir, rdir, ddir in zip(left_image_dirs, right_image_dirs, disparity_dirs):
            l_images = sorted(glob(join(ldir, '*.png')))
            r_images = sorted(glob(join(rdir, '*.png')))
            disparities = sorted(glob(join(ddir, '*.pfm')))
            for i in range(len(disparities)):
                self.image_list += [[l_images[i], r_images[i]]]
                self.disparity_list += [disparities[i]]

        assert len(self.image_list) == len(self.disparity_list)

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

        disparity = frame_utils.read_gen(self.disparity_list[index])

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            disparity = cropper(disparity)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            disparity = cropper(disparity)

        images = np.array(images).transpose(3, 0, 1, 2)
        disparity = disparity.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        disparity = torch.from_numpy(disparity.astype(np.float32))

        return images, disparity

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
        left_image_dirs = sorted([join(f, 'left') for f in image_dirs])
        right_image_dirs = sorted([join(f, 'right') for f in image_dirs])

        disparity_dirs = sorted(glob(join(root, 'disparity/*')))
        disparity_dirs = sorted([join(f, 'left') for f in disparity_dirs])

        assert (len(left_image_dirs) == len(right_image_dirs))
        assert (len(left_image_dirs) == len(disparity_dirs))

        self.image_list = []
        self.disparity_list = []

        for ldir, rdir, ddir in zip(left_image_dirs, right_image_dirs, disparity_dirs):
            l_images = sorted(glob(join(ldir, '*.png')))
            r_images = sorted(glob(join(rdir, '*.png')))
            disparities = sorted(glob(join(ddir, '*.pfm')))
            for i in range(len(disparities)):
                self.image_list += [[l_images[i], r_images[i]]]
                self.disparity_list += [disparities[i]]

        assert len(self.image_list) == len(self.disparity_list)

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

        disparity = frame_utils.read_gen(self.disparity_list[index])

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            disparity = cropper(disparity)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            disparity = cropper(disparity)

        images = np.array(images).transpose(3, 0, 1, 2)
        disparity = disparity.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        disparity = torch.from_numpy(disparity.astype(np.float32))

        return images, disparity

    def __len__(self):
        return self.size * self.replicates

class KITTI(data.Dataset):
    def __init__(self, args, is_cropped=False, root='~/data/data_scene_flow', dstype='training', replicates=1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        l_image_dir = join(root, dstype,'image_2')
        r_image_dir = join(root, dstype, 'image_3')

        disp_dir = join(root, dstype, 'disp_occ_0')

        self.image_list = []
        self.disp_list = []

        images_1 = sorted(glob(join(l_image_dir, '*_10.png')))
        images_2 = sorted(glob(join(r_image_dir, '*_10.png')))
        disps = sorted(glob(join(disp_dir, '*.png')))
        assert len(images_1)==len(images_2)
        assert len(disps)==len(images_1)

        for i in range(len(disps)):
            self.image_list += [[images_1[i], images_2[i]]]
            self.disp_list += [disps[i]]

        assert len(self.image_list) == len(self.disp_list)

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

        disp = frame_utils.read_gen(self.disp_list[index]).astype(float)/256.0

        images = [img1, img2]
        if self.is_cropped:
            cropper = StaticRandomCrop(self.crop_size)
            images = map(cropper, images)
            disp = cropper(disp)
        else:
            cropper = StaticCenterCrop(self.render_size)
            images = map(cropper, images)
            disp = cropper(disp)

        images = np.array(images).transpose(3, 0, 1, 2)
        disp = disp.transpose(2, 0, 1)
        disp = disp[:2,:,:]
        images = torch.from_numpy(images.astype(np.float32))
        disp = torch.from_numpy(disp.astype(np.float32))

        return [images], [disp]

    def __len__(self):
        return self.size * self.replicates

class ImagesFromFolder(data.Dataset):
    def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        left_images = sorted( glob( join(root, 'left/*'+iext) ) )
        right_images = sorted( glob( join(root, 'right/*'+iext) ) )
        self.image_list = []
        for limg,rimg in zip(left_images,right_images):
            self.image_list += [ [ limg, rimg ] ]

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

        return images, torch.zeros(images.size()[0:1] + (1,) + images.size()[-2:])

    def __len__(self):
        return self.size * self.replicates
