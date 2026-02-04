import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)

WANDB_AVAILABLE = False


def inverse_transform_prediction(pred, mat_info, target_size):

    w, h = target_size
    
    try:
        mat = np.array(mat_info, dtype=np.float32)
        transformed = cv2.warpAffine(pred, mat, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderValue=0.)
        return transformed
    except Exception as e:
        logger.warning(f"Matrix transform failed: {e}, using resize fallback")e
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        with amp.autocast(enabled=getattr(args, 'amp', True)):
            if hasattr(model, 'module'):
                result = model.module(image, text, target)
            else:
                result = model(image, text, target)
            
            pred, target_processed, loss = result[0], result[1], result[2]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        if getattr(args, 'max_norm', 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            iou, pr5 = trainMetricGPU(pred, target_processed, 0.35, 0.5)

        if dist.is_initialized():
            dist.all_reduce(loss.detach())
            dist.all_reduce(iou)
            dist.all_reduce(pr5)
            loss = loss / dist.get_world_size()
            iou = iou / dist.get_world_size()
            pr5 = pr5 / dist.get_world_size()


        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        print_freq = getattr(args, 'print_freq', 50)
        if (i + 1) % print_freq == 0:
            progress.display(i + 1)


@torch.no_grad()
def validate(val_loader, model, epoch, args):

    iou_list = []
    model.eval()
    time.sleep(2)
    
    for imgs, texts, params in val_loader:
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)

        if hasattr(model, 'module'):
            preds = model.module(imgs, texts)
        else:
            preds = model(imgs, texts)
        
        preds = torch.sigmoid(preds)

        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        
        for pred, mask_path, mat_info, ori_size in zip(preds, 
                                                       params['mask_path'],
                                                       params['inverse'],
                                                       params['ori_size']):
            h, w = np.array(ori_size)
            pred = pred.cpu().numpy()

            try:
                pred = inverse_transform_prediction(pred, mat_info, (w, h))
                pred = np.array(pred > 0.35)
            except Exception as e:
                logger.warning(f"Transform failed for sample, using resize fallback: {e}")
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
                pred = np.array(pred > 0.35)
            
            mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                continue
            mask = mask / 255.
            
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    
    if len(iou_list) == 0:
        logger.error("No valid samples processed in validation!")
        return 0.0, {}
    
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    
    if dist.is_initialized():
        iou_list = concat_all_gather(iou_list)

    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    
    head = 'Validation: Epoch=[{}/{}]  IoU={:.2f}'.format(
        epoch, args.epochs, 100. * iou.item())
    logger.info(head + temp)
    
    return iou.item(), prec


@torch.no_grad()
def test_qata(test_loader, model, args):
    iou_list = []
    tbar = tqdm(test_loader, desc='Testing QaTa:', ncols=100)
    model.eval()
    time.sleep(2)
    
    if getattr(args, 'visualize', False):
        vis_dir = getattr(args, 'vis_dir', os.path.join(args.output_dir, 'visualizations'))
        os.makedirs(vis_dir, exist_ok=True)
    
    for img, params in tbar:
        img = img.cuda(non_blocking=True)
        
        mask_path = params['mask_path'][0]
        description = params['description'][0]
        ori_size = params['ori_size'][0].numpy()
        inverse_mat = params['inverse'][0]
        img_filename = params['img_filename'][0]
        mask_filename = params['mask_filename'][0]
        
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to load mask: {mask_path}")
            continue
        mask = mask / 255.
        
        if getattr(args, 'visualize', False):
            ori_img = params['ori_img'][0].cpu().numpy()
            img_save_path = os.path.join(vis_dir, f'{img_filename}')
            mask_save_path = os.path.join(vis_dir, f'gt_{mask_filename}')
            cv2.imwrite(img_save_path, cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))

        text = tokenize(description, getattr(args, 'word_len', 77))
        text = text.cuda(non_blocking=True)
        
        if hasattr(model, 'module'):
            pred = model.module(img, text)
        else:
            pred = model(img, text)
        
        pred = torch.sigmoid(pred)

        if pred.shape[-2:] != img.shape[-2:]:
            pred = F.interpolate(pred,
                                 size=img.shape[-2:],
                                 mode='bicubic',
                                 align_corners=True).squeeze()
        
        h, w = ori_size
        pred = pred.cpu().numpy()

        try:
            pred = inverse_transform_prediction(pred, inverse_mat, (w, h))
            pred_binary = np.array(pred > 0.35)
        except Exception as e:
            logger.warning(f"Transform failed, using resize fallback: {e}")
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
            pred_binary = np.array(pred > 0.35)
        
        inter = np.logical_and(pred_binary, mask)
        union = np.logical_or(pred_binary, mask)
        iou = np.sum(inter) / (np.sum(union) + 1e-6)
        iou_list.append(iou)
        
        if getattr(args, 'visualize', False):
            pred_save = (pred * 255).astype(np.uint8)
            pred_binary_save = (pred_binary * 255).astype(np.uint8)
            
            clean_desc = "_".join(description.split()[:5])
            pred_save_path = os.path.join(vis_dir, f'pred_{mask_filename}_iou{iou:.3f}_{clean_desc}.png')
            pred_binary_path = os.path.join(vis_dir, f'pred_binary_{mask_filename}_iou{iou:.3f}.png')
            
            cv2.imwrite(pred_save_path, pred_save)
            cv2.imwrite(pred_binary_path, pred_binary_save)
        
        tbar.set_postfix(IoU=f'{iou:.3f}')
    
    logger.info('=> QaTa Test Results <=')
    iou_list = np.array(iou_list)
    
    prec_list = []
    for thres in np.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).mean()
        prec_list.append(tmp)
    
    mean_iou = iou_list.mean()

    logger.info(f'Mean IoU: {100 * mean_iou:.2f}%')
    for i, thres in enumerate(range(5, 10)):
        logger.info(f'Pr@{thres*10}: {100 * prec_list[i]:.2f}%')

    logger.info(f'IoU Median: {100 * np.median(iou_list):.2f}%')
    logger.info(f'IoU Std: {100 * np.std(iou_list):.2f}%')
    logger.info(f'IoU Min: {100 * np.min(iou_list):.2f}%')
    logger.info(f'IoU Max: {100 * np.max(iou_list):.2f}%')

    results = {
        'mean_iou': mean_iou,
        'median_iou': np.median(iou_list),
        'std_iou': np.std(iou_list),
        'min_iou': np.min(iou_list),
        'max_iou': np.max(iou_list),
        'precision': {}
    }
    
    for i, thres in enumerate(range(5, 10)):
        results['precision'][f'pr@{thres*10}'] = prec_list[i]
    
    return results


def inference(test_loader, model, args):
    return test_qata(test_loader, model, args)