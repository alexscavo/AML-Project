# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class FullModel(nn.Module):

  def __init__(self, model, sem_loss, bd_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, bd_gt, *args, **kwargs):
    
    outputs = self.model(inputs, *args, **kwargs)
    
    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    acc  = self.pixel_acc(outputs[-2], labels)
    loss_s = self.sem_loss(outputs[:-1], labels) # (lambda0*l0 + lambda2*l2)
    loss_b = self.bd_loss(outputs[-1], bd_gt) # boundary loss (l1)

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    try:
        bd_label = torch.where(torch.sigmoid(outputs[-1][:, 0, :, :]) > 0.8, labels, filler) # 0.7
        loss_sb = self.sem_loss([outputs[-2]], bd_label) # BAS-loss (l3)
    except:
        print("Error in loss computation")
        loss_sb = self.sem_loss([outputs[-2]], labels)
    loss = loss_s + loss_b + loss_sb

    return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b] # outputs[:-1] is a list

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)

    if (
        not cfg.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP and 
        not cfg.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR and 
        not cfg.TRAIN.AUGMENTATION.TECHNIQUES.RANDOM_CROP and 
        not cfg.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_NOISE and 
        not cfg.TRAIN.AUGMENTATION.TECHNIQUES.COLOR_JITTER
        ):
        folder_name = "no_aug"
    else:
        folder_name = "aug"
    
    if cfg.TRAIN.DACS.ENABLE:
        folder_name = "dacs"

    if cfg.TRAIN.GAN.ENABLE:
        folder_name = "gan"

    if cfg.TRAIN.AUGMENTATION.ENABLE:
        folder_name+= "_hf" if cfg.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP else ""
        folder_name+= "_gb" if cfg.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR else ""
        folder_name+= "_rc" if cfg.TRAIN.AUGMENTATION.TECHNIQUES.RANDOM_CROP else ""
        folder_name+= "_cj" if cfg.TRAIN.AUGMENTATION.TECHNIQUES.COLOR_JITTER else ""
        folder_name+= "_gn" if cfg.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_NOISE else ""

    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name / folder_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr