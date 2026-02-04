import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List  # 确保包含 List
import numpy as np


class GPUHausdorffLoss(nn.Module):
    """
    GPU优化的Hausdorff距离损失
    统一规范：内部处理sigmoid，避免double sigmoid问题
    """
    
    def __init__(self, 
                 sample_points: int = 1000,
                 reduction: str = 'mean',
                 normalize_by_image_size: bool = True,
                 max_distance_ratio: float = 0.1):
        """
        Args:
            sample_points: 边界采样点数
            reduction: 损失归约方式
            normalize_by_image_size: 是否按图像尺寸归一化
            max_distance_ratio: 最大距离比例
        """
        super().__init__()
        self.sample_points = sample_points
        self.reduction = reduction
        self.normalize_by_image_size = normalize_by_image_size
        self.max_distance_ratio = max_distance_ratio
        
        # Sobel算子用于边界检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def extract_boundary_points(self, mask: torch.Tensor, threshold: float = 0.1) -> list:
        """
        提取边界点坐标（GPU版本）
        
        Args:
            mask: [B, H, W] 概率mask (已经过sigmoid)
            threshold: 边界检测阈值
            
        Returns:
            boundary_points: List of [N_i, 2] 每个batch的边界点坐标
        """
        batch_size, H, W = mask.shape
        
        # 使用Sobel算子检测边界
        mask_4d = mask.unsqueeze(1)
        grad_x = F.conv2d(mask_4d, self.sobel_x, padding=1)
        grad_y = F.conv2d(mask_4d, self.sobel_y, padding=1)
        boundary_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).squeeze(1)
        
        boundary_points_list = []
        
        for b in range(batch_size):
            # 找到边界点
            boundary_mask = boundary_map[b] > threshold
            
            if boundary_mask.sum() == 0:
                # 如果没有边界点，返回中心点
                center_y, center_x = H // 2, W // 2
                points = torch.tensor([[center_y, center_x]], 
                                    device=mask.device, dtype=torch.float32)
            else:
                # 获取边界点坐标
                y_coords, x_coords = torch.where(boundary_mask)
                points = torch.stack([y_coords, x_coords], dim=1).float()
                
                # 如果边界点太多，随机采样
                if points.shape[0] > self.sample_points:
                    indices = torch.randperm(points.shape[0])[:self.sample_points]
                    points = points[indices]
            
            boundary_points_list.append(points)
        
        return boundary_points_list
    
    def compute_hausdorff_distance(self, pred_points: torch.Tensor, 
                                 target_points: torch.Tensor,
                                 image_size: tuple) -> torch.Tensor:
        """计算归一化的Hausdorff距离"""
        if pred_points.shape[0] == 0 or target_points.shape[0] == 0:
            return torch.tensor(0.0, device=pred_points.device)
        
        # 计算距离矩阵 [N, M]
        dist_matrix = torch.cdist(pred_points, target_points, p=2)
        
        # 计算有向Hausdorff距离
        min_dist_pred_to_target = dist_matrix.min(dim=1)[0]  # [N]
        max_min_pred_to_target = min_dist_pred_to_target.max()
        
        min_dist_target_to_pred = dist_matrix.min(dim=0)[0]  # [M]
        max_min_target_to_pred = min_dist_target_to_pred.max()
        
        # Hausdorff距离是两个有向距离的最大值
        hausdorff_dist = torch.max(max_min_pred_to_target, max_min_target_to_pred)
        
        # 关键修复：归一化到合理范围
        if self.normalize_by_image_size:
            H, W = image_size
            diagonal_length = torch.sqrt(torch.tensor(H*H + W*W, dtype=torch.float32, device=hausdorff_dist.device))
            hausdorff_dist = hausdorff_dist / diagonal_length
            hausdorff_dist = torch.clamp(hausdorff_dist, 0, self.max_distance_ratio)
        
        return hausdorff_dist
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Hausdorff距离损失
        
        Args:
            pred_logits: [B, H, W] 预测logits (未经sigmoid)
            target: [B, H, W] 真实mask (0-1)
            
        Returns:
            hausdorff_loss: 标量损失
        """
        # 统一规范：在loss内部执行sigmoid
        pred_probs = torch.sigmoid(pred_logits)
        
        # 获取图像尺寸
        image_size = pred_logits.shape[-2:]
        
        # 提取边界点
        pred_boundary_points = self.extract_boundary_points(pred_probs)
        target_boundary_points = self.extract_boundary_points(target)
        
        # 计算每个batch的Hausdorff距离
        batch_hausdorff_losses = []
        
        for pred_points, target_points in zip(pred_boundary_points, target_boundary_points):
            hd_loss = self.compute_hausdorff_distance(pred_points, target_points, image_size)
            batch_hausdorff_losses.append(hd_loss)
        
        # 堆叠并归约
        hausdorff_losses = torch.stack(batch_hausdorff_losses)
        
        if self.reduction == 'mean':
            return hausdorff_losses.mean()
        elif self.reduction == 'sum':
            return hausdorff_losses.sum()
        else:
            return hausdorff_losses


class UnifiedDiceLoss(nn.Module):
    """
    统一的Dice损失 - 内部处理sigmoid
    """
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, H, W] 预测logits (未经sigmoid)
            target: [B, H, W] 真实标签 (0-1)
        """
        # 统一规范：在loss内部执行sigmoid
        pred_probs = torch.sigmoid(pred_logits)
        
        batch_size = pred_logits.size(0)
        pred_flat = pred_probs.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice_coeff.mean()


class UnifiedBoundaryLoss(nn.Module):
    """
    统一的边界损失 - 内部处理sigmoid
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel算子用于边界检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def compute_boundary_map(self, mask: torch.Tensor) -> torch.Tensor:
        """使用Sobel算子计算边界图"""
        mask_4d = mask.unsqueeze(1)
        grad_x = F.conv2d(mask_4d, self.sobel_x, padding=1)
        grad_y = F.conv2d(mask_4d, self.sobel_y, padding=1)
        boundary_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return boundary_map.squeeze(1)
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, H, W] 预测logits (未经sigmoid)
            target: [B, H, W] 真实标签 (0-1)
        """
        # 统一规范：在loss内部执行sigmoid
        pred_probs = torch.sigmoid(pred_logits)
        
        # 计算边界图
        target_boundary = self.compute_boundary_map(target)
        pred_boundary = self.compute_boundary_map(pred_probs)
        
        # 边界区域权重
        boundary_weight = target_boundary * 2.0 + 1.0
        
        # 计算加权误差
        error = torch.abs(pred_probs - target)
        weighted_error = error * boundary_weight
        
        # 边界一致性损失
        boundary_consistency = F.mse_loss(pred_boundary, target_boundary)
        
        return weighted_error.mean() + 0.1 * boundary_consistency


class UnifiedTripleLoss(nn.Module):
    """
    统一的三重损失函数 (Dice + Boundary + Hausdorff)
    统一规范：所有输入为logits，内部统一处理sigmoid
    """
    
    def __init__(self, 
                 dice_weight: float = 0.5,
                 boundary_weight: float = 0.3,
                 hausdorff_weight: float = 0.2,
                 dice_smooth: float = 1e-5,
                 hausdorff_sample_points: int = 1000,
                 normalize_hausdorff: bool = True):
        """
        Args:
            dice_weight: Dice损失权重
            boundary_weight: Boundary损失权重
            hausdorff_weight: Hausdorff损失权重
            dice_smooth: Dice损失平滑因子
            hausdorff_sample_points: Hausdorff采样点数
            normalize_hausdorff: 是否归一化Hausdorff距离
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.hausdorff_weight = hausdorff_weight
        
        # 初始化各个损失组件
        self.dice_loss_fn = UnifiedDiceLoss(smooth=dice_smooth)
        self.boundary_loss_fn = UnifiedBoundaryLoss()
        self.hausdorff_loss_fn = GPUHausdorffLoss(
            sample_points=hausdorff_sample_points,
            normalize_by_image_size=normalize_hausdorff,
            max_distance_ratio=0.1
        )
        
        print(f"Unified Triple Loss initialized:")
        print(f"  - Dice weight: {dice_weight}")
        print(f"  - Boundary weight: {boundary_weight}")
        print(f"  - Hausdorff weight: {hausdorff_weight}")
        print(f"  - Input format: logits (no external sigmoid needed)")
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, 
                return_components: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pred_logits: [B, H, W] 预测logits (未经sigmoid)
            target: [B, H, W] 真实标签 (0-1)
            return_components: 是否返回各组件损失
            
        Returns:
            total_loss: 标量损失 或 损失字典
        """
        # 计算各个损失组件（都内部处理sigmoid）
        dice_loss = self.dice_loss_fn(pred_logits, target)
        boundary_loss = self.boundary_loss_fn(pred_logits, target)
        hausdorff_loss = self.hausdorff_loss_fn(pred_logits, target)
        
        # 总损失
        total_loss = (self.dice_weight * dice_loss + 
                     self.boundary_weight * boundary_loss + 
                     self.hausdorff_weight * hausdorff_loss)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'dice_loss': dice_loss,
                'boundary_loss': boundary_loss,
                'hausdorff_loss': hausdorff_loss
            }
        else:
            return total_loss


def create_unified_loss(cfg) -> nn.Module:
    """
    根据配置创建统一的损失函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        loss_fn: 统一损失函数实例
    """
    dice_weight = getattr(cfg, 'dice_weight', 0.5)
    boundary_weight = getattr(cfg, 'boundary_weight', 0.3)
    hausdorff_weight = getattr(cfg, 'hausdorff_weight', 0.2)
    dice_smooth = getattr(cfg, 'dice_smooth', 1e-5)
    hausdorff_sample_points = getattr(cfg, 'hausdorff_sample_points', 1000)
    normalize_hausdorff = getattr(cfg, 'normalize_hausdorff', True)
    
    return UnifiedTripleLoss(
        dice_weight=dice_weight,
        boundary_weight=boundary_weight,
        hausdorff_weight=hausdorff_weight,
        dice_smooth=dice_smooth,
        hausdorff_sample_points=hausdorff_sample_points,
        normalize_hausdorff=normalize_hausdorff
    )


# 统一的兼容接口 - 避免double sigmoid
def compute_integrated_loss(pred_logits: torch.Tensor, 
                          gt_masks: torch.Tensor, 
                          cfg) -> torch.Tensor:
    """
    统一的损失计算接口
    
    Args:
        pred_logits: [B, H, W] 预测logits (未经sigmoid)
        gt_masks: [B, H, W] 真实mask (0-1)
        cfg: 配置对象
        
    Returns:
        total_loss: 标量损失
    """
    # 获取权重
    dice_weight = getattr(cfg, 'dice_weight', 0.5)
    boundary_weight = getattr(cfg, 'boundary_weight', 0.3)
    hausdorff_weight = getattr(cfg, 'hausdorff_weight', 0.2)
    dice_smooth = getattr(cfg, 'dice_smooth', 1e-5)
    hausdorff_sample_points = getattr(cfg, 'hausdorff_sample_points', 1000)
    normalize_hausdorff = getattr(cfg, 'normalize_hausdorff', True)
    debug_loss = getattr(cfg, 'debug_loss_components', False)
    
    device = pred_logits.device
    
    # 统一规范：在loss内部执行sigmoid，避免double sigmoid
    pred_probs = torch.sigmoid(pred_logits)
    
    # Dice损失
    batch_size = pred_logits.size(0)
    pred_flat = pred_probs.view(batch_size, -1)
    target_flat = gt_masks.view(batch_size, -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_coeff = (2 * intersection + dice_smooth) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + dice_smooth
    )
    dice_loss = 1 - dice_coeff.mean()
    
    # 边界损失
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    gt_4d = gt_masks.unsqueeze(1)
    grad_x = F.conv2d(gt_4d, sobel_x, padding=1)
    grad_y = F.conv2d(gt_4d, sobel_y, padding=1)
    target_boundary = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).squeeze(1)
    
    pred_4d = pred_probs.unsqueeze(1)
    pred_grad_x = F.conv2d(pred_4d, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_4d, sobel_y, padding=1)
    pred_boundary = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8).squeeze(1)
    
    boundary_weight_map = target_boundary * 2.0 + 1.0
    error = torch.abs(pred_probs - gt_masks)
    weighted_error = error * boundary_weight_map
    boundary_consistency = F.mse_loss(pred_boundary, target_boundary)
    boundary_loss = weighted_error.mean() + 0.1 * boundary_consistency
    
    # Hausdorff损失
    hausdorff_loss_fn = GPUHausdorffLoss(
        sample_points=hausdorff_sample_points,
        normalize_by_image_size=normalize_hausdorff,
        max_distance_ratio=0.1
    ).to(device)
    # 注意：这里传入logits，让Hausdorff内部处理sigmoid
    hausdorff_loss = hausdorff_loss_fn(pred_logits, gt_masks)
    
    # 总损失
    total_loss = (dice_weight * dice_loss + 
                 boundary_weight * boundary_loss + 
                 hausdorff_weight * hausdorff_loss)
    
    # 调试信息
    if debug_loss:
        print(f"UNIFIED LOSS - Dice: {dice_loss.item():.4f}, Boundary: {boundary_loss.item():.4f}, "
              f"Hausdorff: {hausdorff_loss.item():.4f}, Total: {total_loss.item():.4f}")
    
    return total_loss



def extract_boundary_points_for_hd95(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    提取边界点用于HD95计算
    
    Args:
        mask: [H, W] 二值mask
        threshold: 二值化阈值
        
    Returns:
        boundary_points: [N, 2] 边界点坐标 (y, x)
    """
    # 二值化
    binary_mask = (mask > threshold).float()
    
    # 使用形态学操作提取边界
    kernel = torch.ones(3, 3, device=mask.device, dtype=mask.dtype)
    
    # 膨胀和腐蚀
    mask_4d = binary_mask.unsqueeze(0).unsqueeze(0)
    dilated = F.conv2d(mask_4d, kernel.unsqueeze(0).unsqueeze(0), padding=1)
    dilated = (dilated > 0).float()
    
    eroded = -F.conv2d(-mask_4d, kernel.unsqueeze(0).unsqueeze(0), padding=1)
    eroded = (eroded > 8).float()  # 只有当9个像素都是1时才保留
    
    # 边界 = 膨胀 - 腐蚀
    boundary = dilated.squeeze() - eroded.squeeze()
    boundary = (boundary > 0).float()
    
    # 如果没有边界点，返回所有前景点
    if boundary.sum() == 0:
        boundary = binary_mask
    
    # 提取坐标
    y_coords, x_coords = torch.where(boundary > 0)
    if len(y_coords) == 0:
        # 如果仍然没有点，返回图像中心
        H, W = mask.shape
        return torch.tensor([[H//2, W//2]], device=mask.device, dtype=torch.float32)
    
    boundary_points = torch.stack([y_coords.float(), x_coords.float()], dim=1)
    return boundary_points


def compute_hd95(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算95th percentile Hausdorff Distance
    
    Args:
        pred_mask: [H, W] 预测mask (概率或二值)
        gt_mask: [H, W] 真实mask (二值)
        threshold: 二值化阈值
        
    Returns:
        hd95: HD95距离值
    """
    # 提取边界点
    pred_boundary = extract_boundary_points_for_hd95(pred_mask, threshold)
    gt_boundary = extract_boundary_points_for_hd95(gt_mask, threshold)
    
    # 如果任一mask为空，返回最大可能距离
    if pred_boundary.shape[0] == 0 or gt_boundary.shape[0] == 0:
        H, W = pred_mask.shape
        return float(np.sqrt(H*H + W*W))
    
    # 如果边界点相同（完美匹配），返回0
    if torch.equal(pred_boundary, gt_boundary):
        return 0.0
    
    # 计算距离矩阵
    pred_boundary = pred_boundary.float()
    gt_boundary = gt_boundary.float()
    
    # 计算从pred到gt的距离
    dist_pred_to_gt = torch.cdist(pred_boundary, gt_boundary, p=2)  # [N_pred, N_gt]
    min_dist_pred_to_gt = dist_pred_to_gt.min(dim=1)[0]  # [N_pred]
    
    # 计算从gt到pred的距离
    dist_gt_to_pred = torch.cdist(gt_boundary, pred_boundary, p=2)  # [N_gt, N_pred]
    min_dist_gt_to_pred = dist_gt_to_pred.min(dim=1)[0]  # [N_gt]
    
    # 合并所有距离
    all_distances = torch.cat([min_dist_pred_to_gt, min_dist_gt_to_pred])
    
    # 计算95th percentile
    if len(all_distances) == 0:
        return 0.0
    
    # 转换为numpy进行百分位数计算
    distances_np = all_distances.detach().cpu().numpy()
    hd95 = np.percentile(distances_np, 95)
    
    return float(hd95)


def compute_pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算像素精度 (Pixel Accuracy)
    
    Args:
        pred_mask: [H, W] 预测mask (概率或二值)
        gt_mask: [H, W] 真实mask (二值) 
        threshold: 二值化阈值
        
    Returns:
        pa: 像素精度
    """
    # 二值化预测
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()
    
    # 计算正确预测的像素数
    correct_pixels = (pred_binary == gt_binary).float().sum()
    total_pixels = torch.numel(gt_binary)
    
    # 像素精度
    pa = correct_pixels / total_pixels
    return pa.item()


# 计算评估指标（接收概率或logits）- 增强版
def calculate_metrics(pred: torch.Tensor, 
                     gt_mask: torch.Tensor, 
                     threshold: float = 0.5,
                     input_is_logits: bool = True,
                     enable_hd95: bool = True) -> Dict[str, float]:
    """
    计算分割指标（增强版，包含PA和HD95）
    
    Args:
        pred: [H, W] 预测值 (logits或概率)
        gt_mask: [H, W] 真实标签
        threshold: 二值化阈值
        input_is_logits: 输入是否为logits
        compute_hd95: 是否计算HD95（计算量较大）
    
    Returns:
        metrics: 包含所有指标的字典
    """
    if input_is_logits:
        pred_probs = torch.sigmoid(pred)
    else:
        pred_probs = pred
    
    pred_binary = (pred_probs > threshold).float()
    gt_binary = (gt_mask > threshold).float()
    
    # 基础指标
    intersection = (pred_binary * gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    union = pred_sum + gt_sum - intersection
    
    # 避免除零
    epsilon = 1e-7
    
    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * intersection + epsilon) / (pred_sum + gt_sum + epsilon)
    precision = (intersection + epsilon) / (pred_sum + epsilon)
    recall = (intersection + epsilon) / (gt_sum + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # 新增：像素精度 (PA)
    pa = compute_pixel_accuracy(pred_probs, gt_mask, threshold)
    
    # 基础指标字典
    metrics = {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'pa': pa  # 新增：像素精度
    }
    
    # 新增：HD95（可选计算，因为计算量较大）
    if enable_hd95:
        try:
            hd95 = compute_hd95(pred_probs, gt_mask, threshold)
            metrics['hd95'] = hd95
        except Exception as e:
            # 如果HD95计算失败，设为默认值
            metrics['hd95'] = float('inf')
            print(f"Warning: HD95 calculation failed: {e}")
    else:
        metrics['hd95'] = None
    
    return metrics


def calculate_batch_metrics(pred_masks: torch.Tensor, 
                          gt_masks: torch.Tensor, 
                          threshold: float = 0.5,
                          input_is_logits: bool = True,
                          enable_hd95: bool = True) -> Dict[str, float]:
    """
    计算batch的平均指标（增强版，包含PA和HD95）
    
    Args:
        pred_masks: [B, H, W] 预测值 (logits或概率)
        gt_masks: [B, H, W] 真实标签
        threshold: 二值化阈值
        input_is_logits: 输入是否为logits
        compute_hd95: 是否计算HD95
    
    Returns:
        avg_metrics: 平均指标字典
    """
    batch_size = pred_masks.size(0)
    metrics_list = []
    
    for i in range(batch_size):
        metrics = calculate_metrics(
            pred_masks[i], 
            gt_masks[i], 
            threshold, 
            input_is_logits,
            compute_hd95
        )
        metrics_list.append(metrics)
    
    # 计算平均值
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key == 'hd95' and not compute_hd95:
            avg_metrics[key] = None
        else:
            # 过滤掉 None 和 inf 值
            valid_values = [m[key] for m in metrics_list 
                           if m[key] is not None and not np.isinf(m[key])]
            if valid_values:
                avg_metrics[key] = sum(valid_values) / len(valid_values)
            else:
                avg_metrics[key] = 0.0 if key != 'hd95' else float('inf')
    
    return avg_metrics


# 用于大批量验证的高效版本
def calculate_batch_metrics_fast(pred_masks: torch.Tensor, 
                               gt_masks: torch.Tensor, 
                               threshold: float = 0.5,
                               input_is_logits: bool = True) -> Dict[str, float]:
    """
    快速批量指标计算（不包含HD95，用于训练时监控）
    
    Args:
        pred_masks: [B, H, W] 预测值
        gt_masks: [B, H, W] 真实标签
        threshold: 二值化阈值
        input_is_logits: 输入是否为logits
    
    Returns:
        avg_metrics: 平均指标字典（不包含HD95）
    """
    if input_is_logits:
        pred_probs = torch.sigmoid(pred_masks)
    else:
        pred_probs = pred_masks
    
    pred_binary = (pred_probs > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    # 批量计算基础指标
    batch_size = pred_masks.size(0)
    pred_flat = pred_binary.view(batch_size, -1)
    gt_flat = gt_binary.view(batch_size, -1)
    
    intersection = (pred_flat * gt_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    gt_sum = gt_flat.sum(dim=1)
    union = pred_sum + gt_sum - intersection
    
    epsilon = 1e-7
    
    # 批量计算指标
    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * intersection + epsilon) / (pred_sum + gt_sum + epsilon)
    precision = (intersection + epsilon) / (pred_sum + epsilon)
    recall = (intersection + epsilon) / (gt_sum + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # 批量计算像素精度
    correct_pixels = (pred_binary == gt_binary).float().view(batch_size, -1).sum(dim=1)
    total_pixels = gt_binary.view(batch_size, -1).shape[1]
    pa = correct_pixels / total_pixels
    
    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'pa': pa.mean().item(),
        'hd95': None  # 快速版本不计算HD95
    }


# 用于详细分析的完整版本
def calculate_detailed_metrics(pred_masks: torch.Tensor, 
                             gt_masks: torch.Tensor, 
                             threshold_list: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
                             input_is_logits: bool = True) -> Dict[str, Dict[str, float]]:
    """
    计算多个阈值下的详细指标（用于模型分析）
    
    Args:
        pred_masks: [B, H, W] 预测值
        gt_masks: [B, H, W] 真实标签
        threshold_list: 阈值列表
        input_is_logits: 输入是否为logits
    
    Returns:
        detailed_metrics: 多阈值指标字典
    """
    detailed_metrics = {}
    
    for threshold in threshold_list:
        metrics = calculate_batch_metrics(
            pred_masks, gt_masks, 
            threshold=threshold, 
            input_is_logits=input_is_logits,
            compute_hd95=True  # 详细分析时计算HD95
        )
        detailed_metrics[f'threshold_{threshold}'] = metrics
    
    # 找到最佳阈值
    best_threshold = None
    best_dice = 0.0
    for thresh_key, metrics in detailed_metrics.items():
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            best_threshold = thresh_key
    
    detailed_metrics['best_threshold'] = best_threshold
    detailed_metrics['best_metrics'] = detailed_metrics[best_threshold] if best_threshold else None
    
    return detailed_metrics


# 向后兼容的接口
def calculate_metrics_with_pa_hd95(pred: torch.Tensor, 
                                 gt_mask: torch.Tensor, 
                                 threshold: float = 0.5,
                                 input_is_logits: bool = True) -> Dict[str, float]:
    """向后兼容的接口，包含PA和HD95"""
    return calculate_metrics(pred, gt_mask, threshold, input_is_logits, compute_hd95=True)