# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import matplotlib.colors
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


current_index = 0


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')

    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    start = timeit.default_timer()

    # take 50images and run the model and show the results
    images = []
    model.eval()



    for index, batch in enumerate(tqdm(testloader)):
        image, label, _, size, name = batch
        # show normalized image

        # image_norm = image[0].numpy().transpose(1, 2, 0)
        # image_norm = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm))
        # plt.imshow(image_norm)


        # plt.show()
        size = size[0]
        with torch.no_grad():
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image)

            images.append((image[0].numpy().transpose(1, 2, 0), pred[0].numpy(), label[0].numpy()))

        if index == 10:
            break



    # make 3 subplots
    figs, axes = plt.subplots(1, 3)
    img_display = axes[1].imshow(matplotlib.colors.Normalize(clip=False)(images[current_index][0]))
    # pred is (7, 1024, 1024), so we need to convert it to (1024, 1024) to display it: np.argmax(pred, axis=0)
    img_display_2 = axes[2].imshow(np.argmax(images[current_index][1], axis=0), vmin=0, vmax=7, cmap='jet')
    img_display_3 = axes[0].imshow(images[current_index][2], cmap='jet', vmin=0, vmax=7)
    def update_image(step):
        global current_index
        current_index = (current_index + step) % len(images)  # Cycle through images
        img_display.set_array(matplotlib.colors.Normalize(clip=False)(images[current_index][0]))
        img_display_2.set_array(np.argmax(images[current_index][1], axis=0))
        img_display_3.set_array(images[current_index][2])
        figs.canvas.draw()

    def on_key(event):
        if event.key == 'right':  # Next image
            update_image(1)
        elif event.key == 'left':  # Previous image
            update_image(-1)

    figs.canvas.mpl_connect('key_press_event', on_key)
    #
    plt.show()
    # if ('test' in config.DATASET.TEST_SET) and ('city' in config.DATASET.DATASET):
    #     test(config,
    #          test_dataset,
    #          testloader,
    #          model,
    #          sv_dir=final_output_dir)
    #
    # else:
    #     mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
    #                                                        test_dataset,
    #                                                        testloader,
    #                                                        model)
    #
    #     msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
    #         Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
    #                                                 pixel_acc, mean_acc)
    #     logging.info(msg)
    #     logging.info(IoU_array)
    #
    # end = timeit.default_timer()
    # logger.info('Mins: %d' % int((end - start) / 60))
    # logger.info('Done')


if __name__ == '__main__':
    main()
#
#
#
