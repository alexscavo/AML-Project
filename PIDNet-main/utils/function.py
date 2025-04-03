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


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict, targetloader=None):
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

    if targetloader is not None:
        target_iter = iter(targetloader)

    for i_iter, batch in enumerate(trainloader, 0):
        if config.TRAIN.DACS.ENABLE:
            # DACS

            # === SOURCE BATCH ==
            x_s, y_s, bd_s, _, _ = batch
            x_s, y_s, bd_s = x_s.cuda(), y_s.long().cuda(), bd_s.float().cuda()

            # === TARGET BATCH ===
            try:
                x_t, real_gt, _, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(targetloader)
                x_t, _, _, _, _ = next(target_iter)
            x_t = x_t.cuda()

            print(f"Unique labels found in urban image: {y_s.unique().tolist()}")
            print(f"Unique labels found in rural image: {real_gt.unique().tolist()}")
            if 7 in real_gt.unique().tolist() or 7 in y_s.unique().tolist():
                print("Found")
                break

            with torch.no_grad():
                logits_t = model.module.model(x_t)[-2]
                logits_t = torch.nn.functional.interpolate(
                    logits_t, size=x_t.shape[2:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                pseudo_t = torch.argmax(logits_t, dim=1)
                conf_t = torch.softmax(logits_t, dim=1).max(dim=1)[0]
            
            
            # === CONFIDENCE USAGE ==

            # Confidence_mask tells the position of the pixel whose pseudo-labels have 
            # a confidence higher than  the threshold
            confidence_mask = conf_t > config.TRAIN.DACS.THRESHOLD
            
            # Where confidence mask is True put the pixel from pseudo_t, otherwise insert the ignore label (0) 
            pseudo_t_filtered = torch.where(
                confidence_mask,
                pseudo_t,
                torch.tensor(config.TRAIN.IGNORE_LABEL, device=pseudo_t.device)
            )

            # Apply classmix
            x_mixed, y_mixed, source_mask = classmix(x_s, y_s, x_t, pseudo_t_filtered)
            
            # Generate edges from the mixed image
            bd_mixed = generate_safe_edge_map(y_mixed, source_mask, edge_size=3,
                                              edge_pad=True, ignore_label=config.TRAIN.IGNORE_LABEL)
            x_mixed = x_mixed.cuda()
            y_mixed = y_mixed.long().cuda()
            bd_mixed = bd_mixed.float().cuda()

            # === FORWARD PASSES ===
            loss_src, _, acc_src, loss_list_src = model(x_s, y_s, bd_s)
            loss_mix, _, acc_mix, loss_list_mix = model(x_mixed, y_mixed, bd_mixed)

            # === COMBINE LOSSES ===
            lambda_weight = confidence_mask.float().mean().item()
            loss = (loss_src + lambda_weight * loss_mix).mean()
            acc = (acc_src + acc_mix) / 2 # TO BE VERIFIED
            
            sem_loss = loss_list_src[0] + lambda_weight * loss_list_mix[0]
            bce_loss = loss_list_src[1] + lambda_weight * loss_list_mix[1]
        else: 
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
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

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
          model, discriminator, writer_dict, lambda_adv=0.001):

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

    for i_iter, (batch_source, batch_target) in enumerate(zip(trainloader, targetloader)):

        

        images_source, labels, bd_gts, _, _ = batch_source
        images_target, _, _, _, _ = batch_target

        images_source = images_source.cuda()
        images_target = images_target.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        print(f"Labels dtype: {labels.dtype}, shape: {labels.shape}, unique values: {torch.unique(labels)}")
        assert labels.dtype == torch.long, "Labels devono essere di tipo torch.LongTensor"
        assert labels.min() >= 0, f"Labels contengono valori negativi: {labels.min()}"
        assert labels.max() < 8, f"Labels contengono valori >= n_classes: {labels.max()}"

        # 1. Forward seg net per dominio sorgente (supervisionato)
        losses, output_source, acc, loss_list = model(images_source, labels, bd_gts)  
        loss_seg = losses.mean()  

        # 2. Forward seg net per dominio target (non supervisionato)
        output_target = model(images_target)  

        # 3. Discriminatore: distingue tra output sorgente e target
        real_preds = discriminator(output_source.detach())  # No update su G
        fake_preds = discriminator(output_target.detach())

        # 4. Calcolo loss del discriminatore
        loss_D_real = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))
        loss_D_fake = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))
        loss_D = (loss_D_real + loss_D_fake) / 2

        # 5. Update discriminator (senza aggiornare G)
        for param in discriminator.parameters():
            param.requires_grad = False  
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        for param in model.parameters():
            param.requires_grad = True  

        # 6. Update generatore (segmentazione) con loss avversaria
        fake_preds = discriminator(output_target)  # Ora aggiorniamo anche G
        loss_adv = nn.BCEWithLogitsLoss()(fake_preds, torch.ones_like(fake_preds))
        loss_G = loss_seg + lambda_adv * loss_adv  

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Log
        batch_time.update(time.time() - tic)
        tic = time.time()

        ave_loss.update(loss_G.item())
        ave_acc.update(acc.mean().item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer_G, base_lr, num_iters, i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = ('Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {}, '
                   'Loss_G: {:.6f}, Loss_D: {:.6f}, Acc: {:.6f}, Semantic Loss: {:.6f}, BCE Loss: {:.6f}').format(
                      epoch, num_epoch, i_iter, epoch_iters, batch_time.average(),
                      [x['lr'] for x in optimizer_G.param_groups], ave_loss.average(), loss_D.item(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average()
                  )
            logging.info(msg)

    writer.add_scalar('train_loss_G', ave_loss.average(), global_steps)
    writer.add_scalar('train_loss_D', loss_D.item(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def train_adv_multi(config, epoch, num_epoch, epoch_iters, base_lr,
              num_iters, trainloader, targetloader, optimizer_G, 
              optimizer_D1, model, discriminator1, discriminator2,
              writer_dict, lambda_adv1=0.001, lambda_adv2=0.0005):

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

    for i_iter, (batch_source, batch_target) in enumerate(zip(trainloader, targetloader)):

        images_source, labels, bd_gts, _, _ = batch_source
        images_target, _, _, _, _ = batch_target

        images_source = images_source.cuda()
        images_target = images_target.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        # 1. Forward pass della rete di segmentazione (Supervisionato su sorgente)
        losses, output_source_final, output_source_intermediate, acc, loss_list = model(images_source, labels, bd_gts)  
        loss_seg = losses.mean()  

        # 2. Forward pass della rete di segmentazione su target (Non supervisionato)
        output_target_final, output_target_intermediate = model(images_target)  

        # ------------------ TRAINING DISCRIMINATOR 1 (OUTPUT FINALE) ------------------
        real_preds1 = discriminator1(output_source_final.detach())  
        fake_preds1 = discriminator1(output_target_final.detach())

        loss_D1_real = nn.BCEWithLogitsLoss()(real_preds1, torch.ones_like(real_preds1))
        loss_D1_fake = nn.BCEWithLogitsLoss()(fake_preds1, torch.zeros_like(fake_preds1))
        loss_D1 = (loss_D1_real + loss_D1_fake) / 2

        for param in discriminator1.parameters():
            param.requires_grad = True  
        optimizer_D1.zero_grad()
        loss_D1.backward()
        optimizer_D1.step()
        for param in discriminator1.parameters():
            param.requires_grad = False  

        # ------------------ TRAINING DISCRIMINATOR 2 (FEATURE INTERMEDIE) ------------------
        real_preds2 = discriminator2(output_source_intermediate.detach())  
        fake_preds2 = discriminator2(output_target_intermediate.detach())

        loss_D2_real = nn.BCEWithLogitsLoss()(real_preds2, torch.ones_like(real_preds2))
        loss_D2_fake = nn.BCEWithLogitsLoss()(fake_preds2, torch.zeros_like(fake_preds2))
        loss_D2 = (loss_D2_real + loss_D2_fake) / 2

        for param in discriminator2.parameters():
            param.requires_grad = True  
        optimizer_D1.zero_grad()
        loss_D2.backward()
        optimizer_D1.step()
        for param in discriminator2.parameters():
            param.requires_grad = False  

        # ------------------ TRAINING DEL GENERATORE (Segmentazione con Adversarial Loss) ------------------
        fake_preds1 = discriminator1(output_target_final)  # Output finale
        fake_preds2 = discriminator2(output_target_intermediate)  # Output intermedio

        loss_adv1 = nn.BCEWithLogitsLoss()(fake_preds1, torch.ones_like(fake_preds1))
        loss_adv2 = nn.BCEWithLogitsLoss()(fake_preds2, torch.ones_like(fake_preds2))

        loss_G = loss_seg + lambda_adv1 * loss_adv1 + lambda_adv2 * loss_adv2  

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Log
        batch_time.update(time.time() - tic)
        tic = time.time()

        ave_loss.update(loss_G.item())
        ave_acc.update(acc.mean().item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer_G, base_lr, num_iters, i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = ('Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {}, '
                   'Loss_G: {:.6f}, Loss_D1: {:.6f}, Loss_D2: {:.6f}, Acc: {:.6f}, Semantic Loss: {:.6f}, BCE Loss: {:.6f}').format(
                      epoch, num_epoch, i_iter, epoch_iters, batch_time.average(),
                      [x['lr'] for x in optimizer_G.param_groups], ave_loss.average(), loss_D1.item(), loss_D2.item(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average()
                  )
            logging.info(msg)

    writer.add_scalar('train_loss_G', ave_loss.average(), global_steps)
    writer.add_scalar('train_loss_D1', loss_D1.item(), global_steps)
    writer.add_scalar('train_loss_D2', loss_D2.item(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1