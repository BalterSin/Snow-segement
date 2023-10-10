import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image


class Unet(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path points to the weights file in the logs folder
        #   Choose the one with the lower loss
        #   model.pth is the rename of ep100-loss0.021-val_loss0.043.pth
        #-------------------------------------------------------------------#
        "model_path"    : 'model_data/model.pth',
        #--------------------------------#
        #   background + snow = 2
        #--------------------------------#
        "num_classes"   : 2,
        #--------------------------------#
        #   Use resnet50 as the backbone network
        #--------------------------------#
        "backbone"      : "resnet50",
        #--------------------------------#
        #   shape of image
        #--------------------------------#
        "input_shape"   : [512, 512],
        #-------------------------------------------------#
        #   mix_type = 0 represents mixing of the original graph with the generated graph
        #   mix_type = 1 only the generated graph is retained
        #   mix_type = 2 only the background is deducted and only the target in the original image is retained
        #-------------------------------------------------#
        "mix_type"          : 1,
        #--------------------------------#
        #   use Cuda or not
        #   no GPU can be set to False
        #--------------------------------#
        "cuda"          : True,
    }

    #---------------------------------------------------#
    #   Initialization UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   Color setting
        #---------------------------------------------------#
        if self.num_classes <= 2:
            self.colors = [ (0, 0, 0), (128, 0, 0)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   Get the model
        #---------------------------------------------------#
        self.generate()

    #---------------------------------------------------#
    #   Get all the categories
    #---------------------------------------------------#
    def generate(self):
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Image detection
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   Convert the image to RGB image
        #---------------------------------------------------------#
        image       = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   resize
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   Add the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        return pr

        # if self.mix_type == 0:
        #     # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        #     # for c in range(self.num_classes):
        #     #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
        #     #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
        #     #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
        #     seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        #     image   = Image.fromarray(np.uint8(seg_img))
        #     image   = Image.blend(old_img, image, 0.7)

        # elif self.mix_type == 1:
        #     # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        #     # for c in range(self.num_classes):
        #     #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
        #     #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
        #     #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
        #     seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        #     image   = Image.fromarray(np.uint8(seg_img))

        # elif self.mix_type == 2:
        #     seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
        #     image = Image.fromarray(np.uint8(seg_img))
        
        # return image


