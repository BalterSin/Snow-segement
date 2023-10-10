import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    Cuda = True
    num_classes = 2
    backbone    = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Pre-training weights for backbone networks
    #   If model_path is set, the weights do not need to be loaded
    #--------------------------------------------------------------------------------------------------------------------------
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Weights of the whole model
    #   Points to pre-training weights -> model_data/unet_resnet_voc.pth
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "model_data/unet_resnet_voc.pth"

    input_shape = [512, 512]
    
    #----------------------------------------------------#
    #   The training was divided into two phases, the backbone freeze phase and the unfreeze phase.
    #----------------------------------------------------#
    #   Freeze phase training parameters
    #   At this point the backbone of the model is frozen and the feature extraction network does not change
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 30
    Freeze_batch_size   = 32
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #   Unfreeze phase training parameters
    #   At this point the backbone of the model is not frozen anymore and the feature extraction network will change
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16
    Unfreeze_lr         = 1e-5
    #------------------------------#
    #   Dataset path
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #   Whether to use dice loss
    dice_loss       = False
    #---------------------------------------------------------------------# 
    #   Whether to use focal loss to prevent positive and negative sample imbalance
    #---------------------------------------------------------------------# 
    focal_loss      = False
    #---------------------------------------------------------------------# 
    #   Set loss weights
    #---------------------------------------------------------------------# 
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------#
    #   Whether to freeze training or not
    #------------------------------------------------------#
    Freeze_Train    = True
    #------------------------------------------------------#
    #   Used to set whether to use multi-threading to read data
    #------------------------------------------------------#
    num_workers     = 4

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory("logs/")

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
        
    #------------------------------------------------------#
    #   freeze training can speed up training
    #   can also prevent the weights from being corrupted in the early stage of training
    #   Init_Epoch is the starting generation
    #   Interval_Epoch is the freeze training generation
    #   Epoch total training generations
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training")

        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate)


        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)
            lr_scheduler.step()

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training")

        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate)
            
        if Freeze_Train:
            model.unfreeze_backbone()

        for epoch in range(start_epoch,end_epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)
            lr_scheduler.step()
