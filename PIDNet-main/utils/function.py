# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
import torch.nn as nn

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from tools.classmix import classmix, generate_safe_edge_map
import utils.fda as fda


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']


    for i_iter, batch in enumerate(trainloader, 0):
        
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()
        sem_loss = loss_list[0]
        bce_loss = loss_list[1]

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(sem_loss.mean().item())
        avg_bce_loss.update(bce_loss.mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    # Ritorna la loss media per l'epoca
    return ave_loss.average()

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # Discard the first class (index 0)
        IoU_array = IoU_array[1:]

        mean_IoU = IoU_array.mean()
        
        # logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

def train_adv(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, targetloader, optimizer_G, optimizer_D, 
          model, discriminator, writer_dict, lambda_adv=0.0005, iter_size=4):

    # Training mode
    model.train()
    discriminator.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cumulative_loss_G = 0.0
    cumulative_loss_adv = 0.0
    loss_D1_D2_batch = 0.0
    count=0

    for i_iter, (batch_source, batch_target) in enumerate(zip(trainloader, targetloader)):
        
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        cumulative_loss_D = 0.0

        #Train G
        # don't accumulate grads in D
        for param in discriminator.parameters():
            param.requires_grad = False

        images_source, labels, bd_gts, _, _ = batch_source
        images_target, _, _, _, _ = batch_target

        images_source = images_source.to(device)
        images_target = images_target.to(device)
        labels = labels.long().to(device)
        bd_gts = bd_gts.float().to(device)


        # ------------------ TRAINING DEL GENERATORE ------------------
        # 1. Forward seg net per dominio sorgente (supervisionato)
        loss_seg1, output_source, _, _ = model(images_source, labels, bd_gts) #retun delle 3 loss sommate ma unsqueezed
        loss_seg1 = torch.squeeze(loss_seg1, 0).mean()
        loss_seg1.backward()

        cumulative_loss_G += loss_seg1.data.cpu().numpy()

        # Forward pass per il dominio target (adversarial)
        _, output_target, _, _ = model(images_target, labels, bd_gts)
        fake_preds = discriminator(F.softmax(output_target[-1], dim=1))

        bce = nn.BCEWithLogitsLoss()
        loss_adv = bce(fake_preds, torch.zeros_like(fake_preds))
        loss_adv = loss_adv * lambda_adv
        cumulative_loss_G += loss_adv.item()
        loss_adv.backward()

        cumulative_loss_adv += loss_adv.data.cpu().numpy()


        # ------------------ TRAINING DEL DISCRIMINATORE------------------
        for param in discriminator.parameters():
            param.requires_grad = True

        output_source = [t.detach() for t in output_source]
        output_target = [t.detach() for t in output_target]

       
        # train su source
        fake_preds1_d = discriminator(F.softmax(output_source[-1], dim=1)) 
        bce = nn.BCEWithLogitsLoss()
        loss_D_src = bce(fake_preds1_d, torch.zeros_like(fake_preds1_d))
        loss_D_src.backward()

        cumulative_loss_D += loss_D_src.data.cpu().numpy()
     
        
        # train su target
        fake_preds1_d_t = discriminator(F.softmax(output_target[-1], dim=1)) 
        bce = nn.BCEWithLogitsLoss()
        loss_D_trg = bce(fake_preds1_d_t, torch.ones_like(fake_preds1_d_t))
        loss_D_trg.backward()

        cumulative_loss_D += loss_D_trg.data.cpu().numpy()

        loss_D1_D2_batch += cumulative_loss_D

        #--
       
        optimizer_G.step()
        optimizer_D.step()

        # Log
        count+=1
        batch_time.update(time.time() - tic)
        tic = time.time()

        ave_loss.update(cumulative_loss_G)
        ave_acc.update(0)
        avg_sem_loss.update(0)
        avg_bce_loss.update(0)

        lr = adjust_learning_rate(optimizer_G, base_lr, num_iters, i_iter + cur_iters)
        lr = adjust_learning_rate(optimizer_D, base_lr, num_iters, i_iter + cur_iters)
        

        if i_iter % config.PRINT_FREQ == 0:
            msg = ('Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {}, '
                   'Loss_seg: {:.6f}, loss_adv: {:.6f}, loss_D_src: {:.6f}, loss_D_trg: {:.6f}').format(
                      epoch, num_epoch, i_iter, epoch_iters, batch_time.average(),[x['lr'] for x in optimizer_G.param_groups], 
                      loss_seg1, loss_adv, loss_D_src, loss_D_trg
                  )
            logging.info(msg)

    #writer.add_scalar('train_loss_G', ave_loss.average(), global_steps)
    #writer.add_scalar('train_loss_D', cumulative_loss_D, global_steps)
    #writer_dict['train_global_steps'] = global_steps + 1

    # Ritorna la loss media per l'epoca
    final_loss = cumulative_loss_adv + loss_D1_D2_batch + cumulative_loss_G / count
    msg = ('Epoch: [{}/{}], loss_sum: {:.6f}').format(epoch, num_epoch, final_loss)
    logging.info(msg)

    return final_loss


def train_adv_multi(config, epoch, num_epoch, epoch_iters, base_lr,
                    num_iters, trainloader, targetloader, optimizer_G, 
                    optimizer_D1, optimizer_D2, model, discriminator1, discriminator2,
                    writer_dict, lambda_adv1=0.0005, lambda_adv2=0.00049, iter_size=4):

    # Training mode
    model.train()
    discriminator1.train()
    discriminator2.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()

    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    cumulative_loss_G_last = 0.0
    cumulative_loss_adv = 0.0
    loss_D1_D2_batch = 0.0
    count=0

 

    for i_iter, (batch_source, batch_target) in enumerate(zip(trainloader, targetloader)):

        optimizer_G.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()

        cumulative_loss_D1 = 0.0
        cumulative_loss_D2 = 0.0

      
        #Train G
        # don't accumulate grads in D
        for param in discriminator1.parameters():
            param.requires_grad = False
        for param in discriminator2.parameters():
            param.requires_grad = False

        
        images_source, labels, bd_gts, _, _ = batch_source
        images_target, _, _, _, _ = batch_target

        images_source = images_source.to(device)
        images_target = images_target.to(device)
        labels = labels.long().to(device)
        bd_gts = bd_gts.float().to(device)

        # ------------------ TRAINING DEL GENERATORE ------------------
        # 1. Forward seg net per dominio sorgente (supervisionato)
        loss_seg1, outputs_source = model(images_source, labels, bd_gts) #FullModelMulti
        loss_seg1=loss_seg1.mean()
        loss_seg1.backward()

        cumulative_loss_G_last += loss_seg1.data.cpu().numpy()

        

        # Forward pass per il dominio target (adversarial)
        _, outputs_target = model(images_target, labels, bd_gts)
        fake_preds1 = discriminator1(F.softmax(outputs_target[-1], dim=1)) #main
        fake_preds2 = discriminator2(F.softmax(outputs_target[0], dim=1)) #branch P

        bce = nn.BCEWithLogitsLoss()
        loss_adv1 = bce(fake_preds1, torch.zeros_like(fake_preds1)) 
        loss_adv2 = bce(fake_preds2, torch.zeros_like(fake_preds2))

        loss_adv = lambda_adv1 * loss_adv1 + lambda_adv2 * loss_adv2 #ideally lambda_2 smaller than lambda_1
        loss_adv.backward()

        cumulative_loss_adv += loss_adv.data.cpu().numpy()

        # ------------------ TRAINING DEL DISCRIMINATORE------------------

        
        for param in discriminator1.parameters():
                param.requires_grad = True

        for param in discriminator2.parameters():
                param.requires_grad = True

    
        outputs_source = [t.detach() for t in outputs_source]
        outputs_target = [t.detach() for t in outputs_target]

        # train su source
        fake_preds1_d = discriminator1(F.softmax(outputs_source[-1], dim=1)) #main
        fake_preds2_d = discriminator2(F.softmax(outputs_source[0], dim=1)) #branch P

        bce = nn.BCEWithLogitsLoss()
        loss_D1_src = bce(fake_preds1_d, torch.zeros_like(fake_preds1_d)) #here zeros are correct (source_label)
        loss_D2_src = bce(fake_preds2_d, torch.zeros_like(fake_preds2_d))

        loss_D1_src.backward()
        loss_D2_src.backward()
    
        cumulative_loss_D1 += loss_D1_src.data.cpu().numpy()
        cumulative_loss_D2 += loss_D2_src.data.cpu().numpy()
     
        
        # train su target
        fake_preds1_d_t = discriminator1(F.softmax(outputs_target[-1], dim=1)) 
        fake_preds2_d_t = discriminator2(F.softmax(outputs_target[0], dim=1)) 

        bce = nn.BCEWithLogitsLoss()
        loss_D1_trg = bce(fake_preds1_d_t, torch.ones_like(fake_preds1_d_t))
        loss_D2_trg = bce(fake_preds2_d_t, torch.ones_like(fake_preds2_d_t))

        loss_D1_trg.backward()
        loss_D2_trg.backward()


        cumulative_loss_D1 += loss_D1_trg.data.cpu().numpy()
        cumulative_loss_D2 += loss_D2_trg.data.cpu().numpy()

        loss_D1_D2_batch += cumulative_loss_D1 + cumulative_loss_D2

        #--
        optimizer_G.step()
        optimizer_D1.step()
        optimizer_D2.step()

        # Log
        count+=1
        batch_time.update(time.time() - tic)
        tic = time.time()

        lr = adjust_learning_rate(optimizer_G, base_lr, num_iters, i_iter + cur_iters)
        lr = adjust_learning_rate(optimizer_D1, base_lr, num_iters, i_iter + cur_iters)
        lr = adjust_learning_rate(optimizer_D2, base_lr, num_iters, i_iter + cur_iters)


        if i_iter % config.PRINT_FREQ == 0:
            msg = ('Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {}, '
                   'Loss_seg1: {:.6f}, loss_adv1: {:.6f}, loss_adv2: {:.6f},loss_D1_src_trg: {:.6f}, loss_D2_src_trg: {:.6f}').format(
                      epoch, num_epoch, i_iter, epoch_iters, batch_time.average(),[x['lr'] for x in optimizer_G.param_groups], 
                      loss_seg1 ,loss_adv1,loss_adv2, cumulative_loss_D1, cumulative_loss_D2
                  )
            logging.info(msg)

    #writer.add_scalar('train_loss_G', ave_loss.average(), global_steps)
    #writer.add_scalar('train_loss_D1', cumulative_loss_D1, global_steps)
    #writer.add_scalar('train_loss_D2', cumulative_loss_D2, global_steps)
    #writer_dict['train_global_steps'] = global_steps + 1

    # Ritorna la loss media per l'epoca
    final_loss = cumulative_loss_adv + loss_D1_D2_batch + cumulative_loss_G_last/ count
    msg = ('Epoch: [{}/{}], loss_sum: {:.6f}').format(epoch, num_epoch, final_loss)
    logging.info(msg)
    
    return final_loss


def train_FDA(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, targetloader, optimizer, model, writer_dict):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, (batch_source, batch_target) in enumerate(zip(trainloader, targetloader)):

        images_source, labels, bd_gts = batch_source
        images_target, _, _, _, _ = batch_target

        images_source = images_source.to(device)
        images_target = images_target.to(device)
        labels = labels.long().to(device)
        bd_gts = bd_gts.float().to(device)  
    
        losses, _, acc, loss_list = model(images_source, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()
        sem_loss = loss_list[0]
        bce_loss = loss_list[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(sem_loss.mean().item())
        avg_bce_loss.update(bce_loss.mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    # Ritorna la loss media per l'epoca
    return ave_loss.average()