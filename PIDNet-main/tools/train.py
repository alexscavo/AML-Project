# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import albumentations as A
import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
import models.pidnet
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate, train_adv, train_adv_multi, train_FDA
from models.discriminator import FCDiscriminator
from torch import optim
from utils.utils import create_logger, FullModel
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output

import sys
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/loveda/pidnet_small_loveda.yaml", #file di configurazione da usare
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args) #aggiorna config con tutti i parametri trovati nel file di configurazione

    return args

def plot_metrics(train_loss_history, eval_loss_history, mean_iou_history):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(train_loss_history, label='Training Loss', color='blue', marker='o')
    ax[0].plot(eval_loss_history, label='Evaluation Loss', color='orange', marker='x')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(mean_iou_history, label='Mean IoU', color='green', marker='o')
    ax[1].set_title('Mean IoU')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('IoU')
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if torch.cuda.is_available():
        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        gpus = list(config.GPUS)
        if torch.cuda.device_count() != len(gpus):
            print("The gpu numbers do not match!")
            return 0
    gpus = list(config.GPUS)
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED

    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    #crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    crop_size = (1024, 1024)

    train_trasform = None

    if config.TRAIN.AUGMENTATION.ENABLE:
        list_augmentations = []
        if config.TRAIN.AUGMENTATION.TECHNIQUES.RANDOM_CROP:
            list_augmentations.append(A.RandomResizedCrop(1024, 1024, p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP:
            list_augmentations.append(A.HorizontalFlip(p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.COLOR_JITTER:
            list_augmentations.append(A.ColorJitter(p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR:
            list_augmentations.append(A.GaussianBlur(p=0.5))
        if config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_NOISE:
            list_augmentations.append(A.GaussNoise(std_range=(0.2, 0.3), p=0.5))
        if len(list_augmentations) != 0:
            train_trasform = A.Compose(list_augmentations)

    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR,
                        enable_augmentation=True,
                        horizontal_flip=config.TRAIN.AUGMENTATION.TECHNIQUES.HORIZONTAL_FLIP,
                        gaussian_blur=config.TRAIN.AUGMENTATION.TECHNIQUES.GAUSSIAN_BLUR,
                        transform=train_trasform)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)
    
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gpus).cuda() 
    else:
        model = model.to(device)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            print('-'*60)
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    """ # grafici
    plt.ion()  # Modalità interattiva
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # Due grafici: uno per le loss, uno per la mean IoU """
    train_loss_history = []
    eval_loss_history = []
    mean_iou_history = []

    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train_loss=train(config, epoch, config.TRAIN.END_EPOCH, 
                epoch_iters, config.TRAIN.LR, num_iters,
                trainloader, optimizer, model, writer_dict)

        train_loss_history.append(train_loss)

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
            eval_loss_history.append(valid_loss)
            mean_iou_history.append(mean_IoU)

        if flag_rm == 1:
            flag_rm = 0


        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)



    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

    # plt.ioff()
    # plt.show()
if __name__ == '__main__':
    main()