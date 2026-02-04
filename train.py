import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
import glob
from functools import partial
import math

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

import utils.config as config
from utils.dataset import CXRDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn, print_model_info)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='DETRIS Training with Checkpoint Support')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None,
                        help='Override config options')

    parser.add_argument('--local-rank', '--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='Load pretrained weights only')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Manual start epoch (override checkpoint epoch)')
    parser.add_argument('--strict-load', action='store_true',
                        help='Strict mode for weight loading (default: False)')
    
    return parser.parse_args()


def load_config(config_path, opts=None):
    cfg = config.load_cfg_from_cfg_file(config_path)
    if opts:
        cfg = config.merge_cfg_from_list(cfg, opts)

    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if hasattr(v, 'items'):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = {}
    
    for section in ['DATA', 'TRAIN', 'COCOOP', 'CONTRASTIVE', 'LOSS', 'TEST', 'MISC']:
        if hasattr(cfg, section):
            section_cfg = getattr(cfg, section)
            if hasattr(section_cfg, 'items'):
                for key, value in section_cfg.items():
                    flat_config[key] = value
    
    for key, value in flat_config.items():
        setattr(cfg, key, value)

    cfg.config_path = config_path
    
    return cfg


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, cfg=None, strict=True):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    map_location = f'cuda:{cfg.local_rank}' if cfg and hasattr(cfg, 'local_rank') else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    state_dict = checkpoint.get('state_dict', checkpoint)
    
    model_state_dict = {}
    model_has_module = hasattr(model, 'module')
    
    for key, value in state_dict.items():
        if key.startswith('module.') and not model_has_module:
            new_key = key[7:]
            model_state_dict[new_key] = value
        elif not key.startswith('module.') and model_has_module:
            new_key = f'module.{key}'
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=strict)
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        if strict:
            raise
        else:
            logger.warning("Continuing with partial weight loading...")
    
    start_epoch = 0
    best_iou = 0.0
    
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(" Optimizer state loaded")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    if scheduler is not None and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(" Scheduler state loaded")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")

    if scaler is not None and 'scaler' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler'])
            logger.info(" AMP scaler state loaded")
        except Exception as e:
            logger.warning(f"Failed to load scaler state: {e}")
    
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
        logger.info(f" Resume from epoch: {start_epoch}")
    
    if 'best_iou' in checkpoint:
        best_iou = checkpoint['best_iou']
        logger.info(f" Best IoU so far: {best_iou:.4f}")
    
    if 'cur_iou' in checkpoint:
        logger.info(f" Last IoU: {checkpoint['cur_iou']:.4f}")
    
    return start_epoch, best_iou


def load_pretrained_weights(weights_path, model, cfg, strict=False):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    logger.info(f"Loading pretrained weights: {weights_path}")
    
    map_location = f'cuda:{cfg.local_rank}' if hasattr(cfg, 'local_rank') else 'cpu'
    
    if weights_path.endswith('.pth') or weights_path.endswith('.pt'):
        checkpoint = torch.load(weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported weight file format: {weights_path}")
    
    model_state_dict = {}
    model_has_module = hasattr(model, 'module')
    
    for key, value in state_dict.items():
        if key.startswith('module.') and not model_has_module:
            new_key = key[7:]
            model_state_dict[new_key] = value
        elif not key.startswith('module.') and model_has_module:
            new_key = f'module.{key}'
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=strict)
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        logger.info("✓ Pretrained weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        if strict:
            raise
        else:
            logger.warning("Continuing with partial weight loading...")
    
    return 0, 0.0


def setup_environment(cfg):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        cfg.local_rank = local_rank
        cfg.world_size = dist.get_world_size()
        cfg.rank = dist.get_rank()
    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0

    cfg.manual_seed = getattr(cfg, 'manual_seed', 42)
    
    init_random_seed(cfg.manual_seed)
    set_random_seed(cfg.manual_seed)

    if getattr(cfg, 'deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    exp_name = getattr(cfg, 'exp_name', 'QaTa_DETRIS') + timestamp
    cfg.output_dir = os.path.join(cfg.output_folder, exp_name)
    
    if cfg.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        try:
            shutil.copy(cfg.config_path, os.path.join(cfg.output_dir, 'config.yaml'))
        except:
            logger.warning("Failed to backup config file")

    setup_logger(cfg.output_dir, distributed_rank=cfg.rank, filename="train.log")
    
    if dist.is_initialized():
        dist.barrier()
    
    return cfg


def create_data_loaders(cfg):
    logger.info("Creating datasets...")
    
    if not os.path.exists(cfg.data_root):
        raise FileNotFoundError(f"Data root not found: {cfg.data_root}")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU Memory: {gpu_memory:.2f}GB")
    
    train_batch_size = cfg.batch_size
    val_batch_size = getattr(cfg, 'batch_size_val', cfg.batch_size_val)
    workers = getattr(cfg, 'num_workers', 4)
    
    logger.info(f"Using batch_size: {train_batch_size} (from config)")
    logger.info(f"Using val_batch_size: {val_batch_size} (from config)")
    logger.info(f"Using workers: {workers}")
    
    train_dataset = CXRDataset(
        data_root=cfg.data_root,
        split='train',
        mode='train',
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        use_augmentation=getattr(cfg, 'use_augmentation', True),
        aug_probability=getattr(cfg, 'aug_probability', 0.7)
    )
    
    val_dataset = CXRDataset(
        data_root=cfg.data_root,
        split='val',
        mode='val',
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        use_augmentation=False,
        aug_probability=0.0
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if cfg.world_size > 1:
        train_batch_size = train_batch_size // cfg.world_size
        val_batch_size = val_batch_size // cfg.world_size
        logger.info(f"Distributed: batch_size adjusted to {train_batch_size} per GPU")
    
    train_sampler = None
    val_sampler = None
    if cfg.world_size > 1:
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        val_sampler = data.distributed.DistributedSampler(val_dataset, shuffle=False)

    init_fn = partial(worker_init_fn,
                      num_workers=workers,
                      rank=cfg.rank,
                      seed=cfg.manual_seed)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        worker_init_fn=init_fn,
        persistent_workers=False,
        prefetch_factor=2
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
        worker_init_fn=init_fn,
        persistent_workers=False,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, train_sampler


def create_model_and_optimizer(cfg):

    logger.info("Building model...")
    
    try:
        model_result = build_segmenter(cfg)

        if isinstance(model_result, tuple):
            model = model_result[0] 
            logger.info(f"build_segmenter returned tuple with {len(model_result)} elements, using first as model")
            
            if len(model_result) > 1:
                logger.info(f"Additional returns from build_segmenter: {[type(x).__name__ for x in model_result[1:]]}")
        else:
            model = model_result
            logger.info("build_segmenter returned single model")

        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
        
        model = model.cuda()
        logger.info(" Model moved to CUDA")
        
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        logger.error(f"build_segmenter returned: {type(model_result)}")
        raise
    
    if cfg.rank == 0:
        print_model_info(model, cfg)

    if cfg.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=False
        )
        logger.info(" Model wrapped with DistributedDataParallel")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.base_lr,
        betas=(0.9, 0.999),
        weight_decay=getattr(cfg, 'weight_decay', 0.05),
        eps=1e-8
    )
    
    warmup_epochs = getattr(cfg, 'warmup_epochs', 3)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - warmup_epochs,
        eta_min=cfg.base_lr * 0.01
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    scaler = amp.GradScaler(enabled=True)
    
    logger.info(" Optimizer, scheduler, and scaler created")
    
    return model, optimizer, scheduler, scaler


def auto_find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    
    last_checkpoint = os.path.join(output_dir, "last_model.pth")
    if os.path.exists(last_checkpoint):
        return last_checkpoint

    pattern = os.path.join(output_dir, "epoch_*.pth")
    epoch_files = glob.glob(pattern)
    if epoch_files:
        latest_file = max(epoch_files, key=os.path.getmtime)
        return latest_file
    
    return None



def save_checkpoint(model, optimizer, scheduler, scaler, epoch, iou, best_iou, cfg, is_best=False):
    if cfg.rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'cur_iou': iou,
        'best_iou': best_iou,
        'config': cfg.config_path,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    last_path = os.path.join(cfg.output_dir, "last_model.pth")
    torch.save(checkpoint, last_path)
    logger.info(f" Saved last checkpoint: Epoch {epoch}, IoU: {iou:.4f}")
    
    if is_best:
        best_path = os.path.join(cfg.output_dir, "best_model.pth")
        torch.save(checkpoint, best_path)
        logger.info(f" New best model! Epoch {epoch}, IoU: {iou:.4f}")

def main():
    try:
        args = parse_args()
        
        cfg = load_config(args.config, args.opts)

        cfg = setup_environment(cfg)

        if args.resume and args.load_weights:
            logger.warning("Both --resume and --load-weights specified. Using --resume (full recovery)")
            args.load_weights = None

        if not args.resume and not args.load_weights and hasattr(cfg, 'output_folder'):
            possible_dirs = glob.glob(os.path.join(cfg.output_folder, "CF2Seg_*"))
            if possible_dirs:
                latest_dir = sorted(possible_dirs)[-1]
                latest_checkpoint = auto_find_latest_checkpoint(latest_dir)
                if latest_checkpoint:
                    logger.info(f" Auto-detected checkpoint: {latest_checkpoint}")

        if cfg.rank == 0:
            logger.info("="*80)
            logger.info(" Starting Training with Checkpoint Support")

            if args.resume:
                logger.info(f" Resume training from: {args.resume}")
            elif args.load_weights:
                logger.info(f" Load pretrained weights: {args.load_weights}")
            else:
                logger.info(" Training from scratch")
            
            logger.info(f" Data: {cfg.data_root}")
            logger.info(f" Dataset: {getattr(cfg, 'dataset', 'qata_covid19')}")
            logger.info(f" Input size: {cfg.input_size}")
            logger.info(f" Batch size: {cfg.batch_size}")
            logger.info(f" Epochs: {cfg.epochs}")
            logger.info(f" Base LR: {cfg.base_lr}")
            logger.info(f" Optimizer: AdamW (standard)")
            logger.info(f" Scheduler: Warmup + Cosine Annealing")
            logger.info(f" Warmup epochs: {getattr(cfg, 'warmup_epochs', 3)}")
            logger.info(f" Deterministic: {getattr(cfg, 'deterministic', False)}")
            logger.info(f" Benchmark: {getattr(cfg, 'benchmark', True)}")
            logger.info(" Performance Optimizations (Conservative):")
            logger.info(f"  - TF32: {not getattr(cfg, 'deterministic', False)}")
            logger.info("  - Prefetch factor: 2")
            logger.info("  - Persistent workers: False")
            logger.info("  - Optimized AMP: True")
            logger.info("  - Torch compile: Disabled (compatibility)")
            
            if getattr(cfg, 'use_cocoop', False):
                logger.info(" CoCoOp enabled")
            if getattr(cfg, 'use_contrastive', False):
                logger.info(" Contrastive learning enabled")
            
            logger.info("="*80)
        
        train_loader, val_loader, train_sampler = create_data_loaders(cfg)
        
        model, optimizer, scheduler, scaler = create_model_and_optimizer(cfg)

        start_epoch = args.start_epoch
        best_iou = 0.0
        
        if args.resume:
            start_epoch, best_iou = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler, cfg, 
                strict=args.strict_load
            )
            if args.start_epoch > 0:
                start_epoch = args.start_epoch
                logger.info(f" Manual override start_epoch to: {start_epoch}")
                
        elif args.load_weights:
            start_epoch, best_iou = load_pretrained_weights(
                args.load_weights, model, cfg, strict=args.strict_load
            )
            start_epoch = args.start_epoch
            logger.info(" Starting fresh training with pretrained weights")
        
        if start_epoch >= cfg.epochs:
            logger.warning(f"  Start epoch ({start_epoch}) >= total epochs ({cfg.epochs})")
            logger.info(" Training already completed or invalid start epoch")
            return
        
        logger.info(f" Starting training from epoch {start_epoch + 1}...")
        start_time = time.time()
        
        for epoch in range(start_epoch, cfg.epochs):
            epoch_num = epoch + 1
            
            if train_sampler:
                train_sampler.set_epoch(epoch_num)

            if cfg.rank == 0:
                current_lr = optimizer.param_groups[0]['lr']

                progress = epoch_num / cfg.epochs * 100
                
                if epoch_num <= getattr(cfg, 'warmup_epochs', 3):
                    phase = " Warmup"
                else:
                    phase = " Cosine Annealing"
                
                logger.info(f" Epoch {epoch_num}/{cfg.epochs} ({progress:.1f}%) - {phase}")
                logger.info(f" Current LR: {current_lr:.6f}")
            
            try:
                if hasattr(model, 'module') and hasattr(model.module, 'update_epoch'):
                    model.module.update_epoch(epoch_num, cfg.epochs)
                elif hasattr(model, 'update_epoch'):
                    model.update_epoch(epoch_num, cfg.epochs)
            except Exception:
                pass
            
            epoch_start = time.time()
            train(train_loader, model, optimizer, scheduler, scaler, epoch_num, cfg)
            train_time = time.time() - epoch_start
            
            if epoch_num % 3 == 0:
                torch.cuda.empty_cache()

            val_start = time.time()
            iou, prec_dict = validate(val_loader, model, epoch_num, cfg)
            val_time = time.time() - val_start

            if cfg.rank == 0:
                logger.info(f"  Epoch {epoch_num} timing - Train: {train_time:.1f}s, Val: {val_time:.1f}s")

            is_best = iou > best_iou
            if is_best:
                best_iou = iou
            
            save_checkpoint(model, optimizer, scheduler, scaler, epoch_num, iou, best_iou, cfg, is_best)

            scheduler.step()

            if cfg.rank == 0 and epoch_num % 10 == 0:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f" GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                torch.cuda.empty_cache()
        
        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        
        if cfg.rank == 0:
            logger.info("="*80)
            logger.info(" Training completed!")
            logger.info(f" Best IoU: {best_iou:.4f}")
            logger.info(f"  Total time: {total_time}")
            logger.info(f" Output: {cfg.output_dir}")
            logger.info("="*80)
    
    except Exception as e:
        logger.error(f" Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()