import os
import random
import numpy as np
from PIL import Image
from loguru import logger
import sys
import inspect
import torch
from torch import nn
import torch.distributed as dist


def init_random_seed(seed=None, device='cuda', rank=0, world_size=1):
    """Initialize random seed."""
    if seed is not None:
        return seed
    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return tensor
    
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())  # 修复语法错误
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Initialize worker with unique seed."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):  # 修复__init__
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Progress meter for training."""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_logger(output_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """Setup logger."""
    if distributed_rank > 0:
        return
    
    log_file = os.path.join(output_dir, filename)
    
    # 移除现有的handlers
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )
    
    # 添加文件输出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            mode=mode
        )


@torch.no_grad()
def trainMetricGPU(pred, target, thresh_iou=0.35, thresh_prec=0.5):
    """
    Calculate training metrics on GPU.
    
    Args:
        pred: Predicted masks [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W] 
        thresh_iou: Threshold for IoU calculation
        thresh_prec: Threshold for precision calculation
    
    Returns:
        iou: Mean IoU
        prec: Precision at thresh_prec
    """
    # 确保输入是正确的格式
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 应用sigmoid和阈值
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > thresh_iou).float()
    
    # 计算IoU
    intersection = (pred_binary * target).sum(dim=(2, 3))
    union = (pred_binary + target).clamp(0, 1).sum(dim=(2, 3))
    iou = intersection / (union + 1e-6)
    
    # 计算precision（IoU > thresh_prec的比例）
    precision = (iou > thresh_prec).float().mean()
    
    return iou.mean(), precision


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate IoU between prediction and target.
    
    Args:
        pred: Prediction tensor
        target: Ground truth tensor  
        threshold: Threshold for binary prediction
    
    Returns:
        iou: Intersection over Union
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = (pred_binary + target_binary).clamp(0, 1).sum()
    
    iou = intersection / (union + 1e-6)
    return iou


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """Save checkpoint."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(output_dir, 'model_best.pth')
        torch.save(state, best_filepath)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint."""
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        start_epoch = checkpoint.get('epoch', 0)
        best_iou = checkpoint.get('best_iou', 0.0)
        
        # 加载模型状态
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载调度器状态
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return start_epoch, best_iou
    else:
        logger.error(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0


def reduce_tensor(tensor):
    """Reduce tensor across all GPUs."""
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""
    
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.window_size = window_size
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return np.median(self.deque) if self.deque else 0.0

    @property
    def avg(self):
        return np.mean(self.deque) if self.deque else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """Calculate intersection and union for segmentation metrics."""
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def print_model_info(model, input_size=None):
    """Print model information."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = get_model_size(model)
    
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Trainable parameters: {num_trainable_params:,}")
    logger.info(f"  Model size: {model_size:.2f} MB")
    
    if input_size:
        logger.info(f"  Input size: {input_size}")
    
    trainable_ratio = num_trainable_params / num_params * 100
    logger.info(f"  Trainable ratio: {trainable_ratio:.2f}%")