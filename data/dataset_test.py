# Load annotations from lable/test.pkl
# 2021/07/09, Dan

import os
import os.path
import cv2
import glob
import numpy as np
import pickle
import random

import torch
import torch.utils.data as data

from utils.tube_utils import scale_tubes, scale_tubes_abs
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap
from .data_utils import generate_anchors

WIDTH, HEIGHT = 400, 400
TEM_REDUCE = 4    # 4 for I3D backbone

class DatasetTest(data.Dataset):
    """
    Customized Dataset to generate input sequences from test.pkl
    """
    def __init__(self, path_label, T=3, chunks=3, target_fps=12, transform=None, stride=1, anchor_mode="1", im_format='frame%04d.jpg'):
        """
        Arguments:
            data_root: path to videos
            T: input sequence length
            target_fps: target frame rate for our model
            transform: input transformation
            stride: stride
            anchor_mode: anchor_mode used
            im_format: format used to load frames
        """

        self.data_root = "/data/CLASP-DATA/CLASP2-STEP/data/frames"
        self.path_label= path_label
        self.T = T
        self.chunks = chunks
        self.target_fps = target_fps
        self.transform = transform
        self.stride = stride
        self.anchor_mode = anchor_mode
        self.im_format = im_format

        self.make_list()
        print('Datalist len: ', len(self.data))
    
    def make_list(self):
        
        with open(self.path_label,'rb') as fin:
            annots = pickle.load(fin)
        
        # Use Video names to add every second as fid for each video name
        self.data=[]
        for name_video in sorted(annots.keys()):
            path_video = os.path.join(self.data_root, name_video)
            path_secs = glob.glob(path_video+'/*')
            num_secs = len(path_secs)-1
            for path_sec in sorted(path_secs):
                self.data.append((path_video, int(path_sec.split('/')[-1]), num_secs))

    def read_images(self, path_video, sid, num_secs):
        """
        Load images from disk for middle frame fid

        return:
            an array with shape (T,H,W,C)
        """

        # set stride according to source fps and target fps
        T = self.T * self.chunks * TEM_REDUCE
        images = []

        # left of middel frame
        num_left = int(T/2)
        i = 1
        while num_left > 0:
            img_path = os.path.join(path_video, '{:05d}'.format(max(0,sid-i)))
            num_get = min(num_left, self.target_fps)
            images.extend(self._load_images(img_path, num=num_get, fps=self.target_fps, direction='backward'))

            num_left -= self.target_fps
            i += 1
        # reverse list
        images = images[::-1]

        # right of middel frame
        num_right = int(np.ceil(T/2))
        i = 0
        while num_right > 0:
            img_path = os.path.join(path_video, '{:05d}'.format(min(sid+i, num_secs)))
            num_get = min(num_right, self.target_fps)
            images.extend(self._load_images(img_path, num=num_get, fps=self.target_fps, direction='forward'))

            num_right -= self.target_fps
            i += 1

        return np.stack(images, axis=0)
    
    def _load_images(self, path, num, fps=12, direction='forward'):
        """
        Load images in a folder with given num and fps, direction can be either 'forward' or 'backward'
        """

        img_names = glob.glob(os.path.join(path, '*.jpg'))
        if len(img_names) == 0:
            img_names = glob.glob(os.path.join(path, '*.png'))
            if len(img_names) == 0:
                raise ValueError("Image path {} not Found".format(path))
            # raise ValueError("Image path {} not Found".format(path))
        img_names = sorted(img_names)

        # resampling according to fps
        index = np.linspace(0, len(img_names), fps, endpoint=False, dtype=np.int)
        if direction == 'forward':
            index = index[:num]
        elif direction == 'backward':
            index = index[-num:][::-1]
        else:
            raise ValueError("Not recognized direction", direction)

        images = []
        for idx in index:
            img_name = img_names[idx]
            if os.path.isfile(img_name):
                img = cv2.imread(img_name)
                images.append(img)
            else:
                raise ValueError("Image not found!", img_name)

        return images


    def __getitem__(self, index):
        """
        Return:
            images: FloatTensor, shape [T, C, H, W]
            anchors: FloatTensor, shape [num, self.T, 4]
        """

        # pull an example sequence
        path_vid, sid, num_secs = self.data[index]

        # load data
        images = self.read_images(path_vid, sid, num_secs)
        # data augmentation
        images, _,_ = self.transform(images)

        # BGR to RGB (for opencv)
        images = images[:, :, :, (2,1,0)]
        # swap dimensions to [T, C, W, H]
        images = torch.from_numpy(images).permute(0,3,1,2)

        # get anchor tubes
        if self.anchor_mode == "0":
            anchors = generate_anchors([4/3], [5/6])
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "1":
            anchors = generate_anchors([4/3,2], [5/6,3/4]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "2":
            anchors = generate_anchors([4/3,2,3], [5/6,3/4,1/2]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "3":
            anchors = generate_anchors([4/3,2,3,4], [5/6,3/4,1/2,1/4]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "4":
            anchors = generate_anchors([4/3,2,3,4,5], [5/6,3/4,1/2,1/4,0]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        else:    # void anchor
            anchor_tubes = np.zeros([1,self.T,4])

        # rescale tubes to absolute position
        anchor_tubes = scale_tubes_abs(anchor_tubes, WIDTH, HEIGHT)

        # collect useful information
        info = {'video_name': path_vid, 'fid': sid}

        return images, anchor_tubes, info

    def __len__(self):
        return len(self.data)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images, tubes and anchors
       We use a list of tensors for tubes and anchors since they may have different sizes for each sequence
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) proposal tubes for a given image are stacked on 0 dim
            3) (list of dict) informations
    """

    imgs = []
    tubes = []
    infos = []
    for sample in batch:
        imgs.append(sample[0])
        tubes.append(sample[1])
        infos.append(sample[2])

    if imgs[0] is not None:
        imgs = torch.stack(imgs, 0)

    return imgs, tubes, infos