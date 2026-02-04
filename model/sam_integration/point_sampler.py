# -*- coding: utf-8 -*-
"""
Gumbel-Softmax Point Sampler (支持训练阶段soft采样)
修复版本：处理空掩码和边界情况
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GumbelPointSampler(nn.Module):
    """
    使用Gumbel-Softmax从概率分布中采样点坐标
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        hard: bool = True,
        num_points: int = 1,
        min_confidence: float = 0.4
    ):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.num_points = num_points
        self.min_confidence = min_confidence
    
    def _sample_gumbel(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        uniform = torch.rand(shape, device=device)
        return -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
    
    def _gumbel_softmax_sample(self, logits: torch.Tensor) -> torch.Tensor:
        gumbel_noise = self._sample_gumbel(logits.shape, logits.device)
        y = (logits + gumbel_noise) / self.temperature
        y_soft = F.softmax(y, dim=-1)
    
        if self.hard:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        
            # 修复：在训练模式下不使用detach，保持梯度流
            if self.training:
                # 训练时：保持完整梯度，使用soft版本
                y = y_soft
            else:
                # 推理时：使用hard版本的straight-through estimator
                y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
    
        return y
    
    def _mask_to_prob_dist(self, mask: torch.Tensor) -> torch.Tensor:
        """将掩码转换为概率分布，添加边界情况处理"""
        B, _, H, W = mask.shape
        prob_mask = torch.sigmoid(mask)
        
        # 应用置信度阈值
        prob_mask = torch.where(prob_mask > self.min_confidence, prob_mask, torch.zeros_like(prob_mask))
        prob_dist = prob_mask.view(B, -1)
        
        # 计算每个样本的总概率
        total_prob = prob_dist.sum(dim=-1, keepdim=True)
        
        # 检查是否有有效概率
        valid_mask = (total_prob > 1e-8).squeeze(-1)  # [B]
        
        # 对于没有有效概率的样本，使用均匀分布
        uniform_dist = torch.ones_like(prob_dist) / (H * W)
        prob_dist = torch.where(valid_mask.unsqueeze(-1), 
                               prob_dist / (total_prob + 1e-8), 
                               uniform_dist)
        
        return prob_dist, valid_mask
    
    def _generate_fallback_points(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """为无效掩码生成后备点"""
        # 在图像中心区域生成随机点
        center_x, center_y = W // 2, H // 2
        radius = min(W, H) // 4
        
        fallback_coords = []
        for _ in range(self.num_points):
            # 在中心区域随机采样
            x = torch.randint(max(0, center_x - radius), 
                            min(W, center_x + radius + 1), 
                            (B,), device=device).float()
            y = torch.randint(max(0, center_y - radius), 
                            min(H, center_y + radius + 1), 
                            (B,), device=device).float()
            coords = torch.stack([x, y], dim=-1)  # [B, 2]
            fallback_coords.append(coords)
        
        return torch.stack(fallback_coords, dim=1)  # [B, num_points, 2]
    
    def forward(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = mask.shape
        device = mask.device
        
        # 检查输入是否有效
        if B == 0:
            raise ValueError("Batch size cannot be 0")
        
        # 转换为概率分布并获取有效性标记
        prob_dist, valid_mask = self._mask_to_prob_dist(mask)
        logits = torch.log(prob_dist + 1e-8)

        # 如果所有样本都无效，生成后备点
        if not valid_mask.any():
            print(f"Warning: All masks are invalid (below confidence {self.min_confidence}), using fallback points")
            fallback_points = self._generate_fallback_points(B, H, W, device)
            point_labels = torch.ones(B, self.num_points, device=device, dtype=torch.long)
            return fallback_points.float(), point_labels

        all_coords = []
        
        try:
            for point_idx in range(self.num_points):
                if self.hard:
                    one_hot = self._gumbel_softmax_sample(logits)
                    indices = one_hot.argmax(dim=-1)
                    coords_x = (indices % W).float()
                    coords_y = (indices // W).float()
                else:
                    prob = F.softmax(logits / self.temperature, dim=-1)
                    y_grid, x_grid = torch.meshgrid(
                        torch.arange(H, device=device),
                        torch.arange(W, device=device),
                        indexing='ij'
                    )
                    x_grid = x_grid.flatten().float()
                    y_grid = y_grid.flatten().float()
                    coords_x = (prob * x_grid).sum(dim=-1)
                    coords_y = (prob * y_grid).sum(dim=-1)
                
                coords = torch.stack([coords_x, coords_y], dim=-1)  # [B, 2]
                all_coords.append(coords)
                
                # 为下一个点更新logits（避免重复采样）
                if self.num_points > 1 and self.hard and point_idx < self.num_points - 1:
                    logits = logits.scatter_(-1, indices.unsqueeze(-1), float('-inf'))
        
        except Exception as e:
            print(f"Error during point sampling: {e}")
            print(f"Mask shape: {mask.shape}, valid_mask: {valid_mask.sum()}/{B}")
            print(f"Prob_dist stats: min={prob_dist.min():.6f}, max={prob_dist.max():.6f}, sum={prob_dist.sum(dim=-1)}")
            
            # 生成后备点
            fallback_points = self._generate_fallback_points(B, H, W, device)
            point_labels = torch.ones(B, self.num_points, device=device, dtype=torch.long)
            return fallback_points.float(), point_labels

        if len(all_coords) == 0:
            print("Warning: No points sampled, using fallback points")
            fallback_points = self._generate_fallback_points(B, H, W, device)
            point_labels = torch.ones(B, self.num_points, device=device, dtype=torch.long)
            return fallback_points.float(), point_labels

        point_coords = torch.stack(all_coords, dim=1)  # [B, num_points, 2]
        point_labels = torch.ones(B, self.num_points, device=device, dtype=torch.long)
        
        # 对无效样本使用后备点
        if not valid_mask.all():
            invalid_indices = ~valid_mask
            if invalid_indices.any():
                fallback_points = self._generate_fallback_points(invalid_indices.sum().item(), H, W, device)
                point_coords[invalid_indices] = fallback_points
                print(f"Used fallback points for {invalid_indices.sum()} invalid samples")
        
        return point_coords.float(), point_labels
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def get_config(self) -> dict:
        return {
            'temperature': self.temperature,
            'hard': self.hard,
            'num_points': self.num_points,
            'min_confidence': self.min_confidence
        }
    
    def diagnose_mask(self, mask: torch.Tensor) -> dict:
        """诊断掩码质量的辅助函数"""
        B, _, H, W = mask.shape
        prob_mask = torch.sigmoid(mask)
        
        stats = {
            'batch_size': B,
            'mask_shape': (H, W),
            'mask_min': mask.min().item(),
            'mask_max': mask.max().item(),
            'mask_mean': mask.mean().item(),
            'prob_min': prob_mask.min().item(),
            'prob_max': prob_mask.max().item(),
            'prob_mean': prob_mask.mean().item(),
            'valid_pixels_per_sample': (prob_mask > self.min_confidence).sum(dim=(1,2,3)).cpu().tolist(),
            'min_confidence': self.min_confidence
        }
        
        return stats