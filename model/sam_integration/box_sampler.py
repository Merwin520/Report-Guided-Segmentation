# -*- coding: utf-8 -*-
"""
CDF-based Box Sampler (基于累积分布函数的可微分Box提取)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CDFBoxSampler(nn.Module):
    """
    使用累积分布函数(CDF)从mask中提取bounding box
    完全可微分，基于统计分位数方法
    """
    
    def __init__(
        self,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
        temperature: float = 10.0,
        min_confidence: float = 0.01,
        epsilon: float = 1e-8
    ):
        """
        Args:
            quantile_low: 下分位数，控制box的紧密程度 (0-1)
            quantile_high: 上分位数，控制box的紧密程度 (0-1)  
            temperature: 温度参数，控制sigmoid的陡峭程度
            min_confidence: 最小置信度阈值，过滤噪声
            epsilon: 数值稳定性参数
        """
        super().__init__()
        
        assert 0 <= quantile_low < quantile_high <= 1, "Invalid quantile range"
        
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.epsilon = epsilon
    
    def _mask_to_prob_dist(self, mask: torch.Tensor) -> torch.Tensor:
        """将mask转换为概率分布"""
        # 应用温度控制的sigmoid
        prob_mask = torch.sigmoid(mask * self.temperature)
        
        # 过滤低置信度区域
        prob_mask = torch.where(
            prob_mask > self.min_confidence, 
            prob_mask, 
            torch.zeros_like(prob_mask)
        )
        
        return prob_mask
    
    def _compute_marginal_distributions(self, prob_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算边际分布"""
        B, _, H, W = prob_mask.shape
        
        # 计算边际概率
        marginal_y = prob_mask.sum(dim=-1)  # [B, 1, H] - 每行的概率和
        marginal_x = prob_mask.sum(dim=-2)  # [B, 1, W] - 每列的概率和
        
        # 归一化为概率分布
        marginal_y = marginal_y / (marginal_y.sum(dim=-1, keepdim=True) + self.epsilon)
        marginal_x = marginal_x / (marginal_x.sum(dim=-1, keepdim=True) + self.epsilon)
        
        return marginal_y.squeeze(1), marginal_x.squeeze(1)  # [B, H], [B, W]
    
    def _differentiable_quantile(self, distribution: torch.Tensor, quantile: float) -> torch.Tensor:
        """
        可微分分位数计算
        使用软排序和线性插值实现
        
        Args:
            distribution: [B, N] 概率分布
            quantile: 目标分位数 (0-1)
        
        Returns:
            quantile_values: [B] 分位数对应的索引值(连续)
        """
        B, N = distribution.shape
        device = distribution.device
        
        # 计算累积分布函数
        cdf = torch.cumsum(distribution, dim=-1)  # [B, N]
        
        # 创建位置索引
        positions = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)  # [B, N]
        
        # 找到目标分位数在CDF中的位置
        target_cdf = torch.full((B, 1), quantile, device=device)  # [B, 1]
        
        # 计算CDF与目标分位数的距离
        cdf_diff = torch.abs(cdf - target_cdf)  # [B, N]
        
        # 使用softmax作为软选择机制（温度控制精度）
        soft_selection_temp = 50.0  # 较高温度获得更精确的选择
        weights = F.softmax(-cdf_diff * soft_selection_temp, dim=-1)  # [B, N]
        
        # 加权平均得到连续的分位数位置
        quantile_positions = (weights * positions).sum(dim=-1)  # [B]
        
        return quantile_positions
    
    def _extract_bbox_from_marginals(
        self, 
        marginal_y: torch.Tensor, 
        marginal_x: torch.Tensor
    ) -> torch.Tensor:
        """从边际分布中提取bounding box坐标"""
        
        # 计算Y方向的分位数
        y_min_pos = self._differentiable_quantile(marginal_y, self.quantile_low)
        y_max_pos = self._differentiable_quantile(marginal_y, self.quantile_high)
        
        # 计算X方向的分位数  
        x_min_pos = self._differentiable_quantile(marginal_x, self.quantile_low)
        x_max_pos = self._differentiable_quantile(marginal_x, self.quantile_high)
        
        # 组装bounding box [x_min, y_min, x_max, y_max]
        bboxes = torch.stack([x_min_pos, y_min_pos, x_max_pos, y_max_pos], dim=-1)  # [B, 4]
        
        return bboxes
    
    def _clamp_bbox_to_valid_range(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """将bbox坐标限制在有效范围内"""
        # 限制坐标范围
        bboxes = torch.clamp(bboxes, 0, max(H-1, W-1))
        
        # 确保 x_min < x_max, y_min < y_max
        x_min, y_min, x_max, y_max = bboxes.split(1, dim=-1)
        
        # 防止box退化为点或负宽高
        min_size = 1.0
        x_max = torch.maximum(x_max, x_min + min_size)
        y_max = torch.maximum(y_max, y_min + min_size)
        
        # 再次限制范围
        x_max = torch.clamp(x_max, 0, W-1)
        y_max = torch.clamp(y_max, 0, H-1)
        
        return torch.cat([x_min, y_min, x_max, y_max], dim=-1)
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        从mask中提取bounding box
        
        Args:
            mask: [B, 1, H, W] input mask tensor
            
        Returns:
            bboxes: [B, 4] bounding boxes in format [x_min, y_min, x_max, y_max]
        """
        B, C, H, W = mask.shape
        
        if C != 1:
            raise ValueError(f"Expected single channel mask, got {C} channels")
        
        # 1. 转换为概率分布
        prob_mask = self._mask_to_prob_dist(mask)
        
        # 2. 计算边际分布
        marginal_y, marginal_x = self._compute_marginal_distributions(prob_mask)
        
        # 3. 基于CDF提取bbox
        bboxes = self._extract_bbox_from_marginals(marginal_y, marginal_x)
        
        # 4. 限制到有效范围
        bboxes = self._clamp_bbox_to_valid_range(bboxes, H, W)
        
        return bboxes
    
    def set_quantiles(self, quantile_low: float, quantile_high: float):
        """动态调整分位数参数"""
        assert 0 <= quantile_low < quantile_high <= 1, "Invalid quantile range"
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        print(f"Box quantiles updated: low={quantile_low}, high={quantile_high}")
    
    def set_temperature(self, temperature: float):
        """调整温度参数"""
        self.temperature = temperature
        print(f"Box sampler temperature updated: {temperature}")
    
    def get_config(self) -> dict:
        """获取配置参数"""
        return {
            'quantile_low': self.quantile_low,
            'quantile_high': self.quantile_high,
            'temperature': self.temperature,
            'min_confidence': self.min_confidence,
            'epsilon': self.epsilon
        }
    
    def check_gradient_flow(self, mask: torch.Tensor) -> bool:
        """检查梯度流是否正常"""
        if not mask.requires_grad:
            mask = mask.requires_grad_(True)
        
        # 前向传播
        bboxes = self.forward(mask)
        
        if not bboxes.requires_grad:
            print("❌ 梯度断开：输出不需要梯度")
            return False
        
        try:
            # 模拟反向传播
            loss = bboxes.sum()
            loss.backward()
            
            if mask.grad is None:
                print("❌ 梯度断开：输入没有梯度")
                return False
            
            print("✅ Box Sampler梯度流正常")
            return True
            
        except Exception as e:
            print(f"❌ 反向传播失败: {e}")
            return False


class AdaptiveBoxSampler(CDFBoxSampler):
    """
    自适应Box采样器
    根据mask的特性动态调整参数
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 自适应参数
        self.auto_adjust_quantiles = True
        self.min_quantile_gap = 0.1  # 最小分位数间隔
        self.max_quantile_gap = 0.9  # 最大分位数间隔
    
    def _analyze_mask_properties(self, prob_mask: torch.Tensor) -> dict:
        """分析mask特性"""
        B, _, H, W = prob_mask.shape
        
        # 计算mask的统计特性
        total_prob = prob_mask.sum(dim=(-1, -2))  # [B]
        mask_density = total_prob / (H * W)  # 密度
        
        # 计算质心
        y_coords = torch.arange(H, device=prob_mask.device, dtype=torch.float32).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=prob_mask.device, dtype=torch.float32).view(1, 1, 1, W)
        
        centroid_y = (prob_mask * y_coords).sum(dim=(-1, -2)) / (total_prob + self.epsilon)
        centroid_x = (prob_mask * x_coords).sum(dim=(-1, -2)) / (total_prob + self.epsilon)
        
        # 计算分散度
        variance_y = (prob_mask * (y_coords - centroid_y.view(B, 1, 1, 1))**2).sum(dim=(-1, -2)) / (total_prob + self.epsilon)
        variance_x = (prob_mask * (x_coords - centroid_x.view(B, 1, 1, 1))**2).sum(dim=(-1, -2)) / (total_prob + self.epsilon)
        
        spread = torch.sqrt(variance_y + variance_x)
        
        return {
            'density': mask_density,
            'spread': spread,
            'centroid': (centroid_x, centroid_y)
        }
    
    def _adaptive_quantile_adjustment(self, mask_properties: dict) -> Tuple[float, float]:
        """根据mask特性自适应调整分位数"""
        density = mask_properties['density'].mean().item()
        spread = mask_properties['spread'].mean().item()
        
        # 根据密度和分散度调整分位数
        if density < 0.1:  # 稀疏mask，使用更宽松的分位数
            gap = min(self.max_quantile_gap, 0.8)
        elif density > 0.5:  # 密集mask，使用更紧的分位数
            gap = max(self.min_quantile_gap, 0.2)
        else:  # 中等密度，使用默认分位数
            gap = self.quantile_high - self.quantile_low
        
        # 确保分位数对称
        center = 0.5
        quantile_low = max(0.01, center - gap/2)
        quantile_high = min(0.99, center + gap/2)
        
        return quantile_low, quantile_high
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """自适应前向传播"""
        
        # 分析mask特性
        prob_mask = self._mask_to_prob_dist(mask)
        
        if self.auto_adjust_quantiles:
            mask_properties = self._analyze_mask_properties(prob_mask)
            orig_low, orig_high = self.quantile_low, self.quantile_high
            
            # 动态调整分位数
            self.quantile_low, self.quantile_high = self._adaptive_quantile_adjustment(mask_properties)
            
            # 继续正常流程
            result = super().forward(mask)
            
            # 恢复原始分位数
            self.quantile_low, self.quantile_high = orig_low, orig_high
            
            return result
        else:
            return super().forward(mask)


# 工厂函数
def create_box_sampler(sampler_type: str = 'cdf', **kwargs) -> nn.Module:
    """
    工厂函数：创建不同类型的box sampler
    
    Args:
        sampler_type: 'cdf' 或 'adaptive'
        **kwargs: sampler的配置参数
    
    Returns:
        box sampler实例
    """
    if sampler_type == 'cdf':
        return CDFBoxSampler(**kwargs)
    elif sampler_type == 'adaptive':
        return AdaptiveBoxSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Choose from 'cdf', 'adaptive'")