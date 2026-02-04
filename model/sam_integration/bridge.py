import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any
import warnings
import math

from .box_sampler import create_box_sampler
BOX_SAMPLER_AVAILABLE = True

class TrainableBridge(nn.Module):
    """
    端到端可训练的DETRIS-SAM桥梁
    使用纯PyTorch操作保证梯度流完整性
    新增：Soft morphological dilation 和 Box sampling 支持
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        use_sigmoid: bool = True,
        apply_threshold: bool = False,
        threshold_value: float = 0.5,
        smooth_kernel_size: int = 3,
        device: Optional[torch.device] = None,
        use_grid_sample: bool = True,
        use_soft_dilation: bool = False,
        dilation_kernel_size: int = 3,
        dilation_iterations: int = 1,
        dilation_beta: float = 5.0,
        dilation_strength: float = 1.0,
        # 新增 box sampling 参数
        use_box_sampling: bool = False,
        box_sampler_type: str = 'cdf',
        box_sampler_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.target_size = target_size
        self.use_sigmoid = use_sigmoid
        self.apply_threshold = apply_threshold
        self.threshold_value = threshold_value
        self.smooth_kernel_size = smooth_kernel_size
        self.use_grid_sample = use_grid_sample
        
        # Soft morphological dilation 参数
        self.use_soft_dilation = use_soft_dilation
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations = dilation_iterations
        self.dilation_beta = dilation_beta
        self.dilation_strength = dilation_strength
        
        # Box sampling 参数
        self.use_box_sampling = use_box_sampling and BOX_SAMPLER_AVAILABLE
        if self.use_box_sampling:
            if box_sampler_config is None:
                box_sampler_config = {}
            self.box_sampler = create_box_sampler(box_sampler_type, **box_sampler_config)
        else:
            self.box_sampler = None
            
        # 如果用户想使用box sampling但模块不可用，给出警告
        if use_box_sampling and not BOX_SAMPLER_AVAILABLE:
            print("Warning: Box sampling requested but box_sampler module not available")
        
        # 保存设备信息
        self._device = device
        
        # 平滑滤波器（固定权重）
        if smooth_kernel_size > 1:
            smooth_kernel = self._create_smooth_kernel(smooth_kernel_size)
            self.register_buffer('smooth_kernel', smooth_kernel)
        else:
            self.smooth_kernel = None
            
        # 设备管理
        if device is not None:
            self.to(device)
    
    @property
    def device(self) -> torch.device:
        """获取当前设备"""
        if self._device is not None:
            return self._device
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device('cpu')
    
    def _create_smooth_kernel(self, kernel_size: int) -> torch.Tensor:
        """创建高斯平滑核"""
        sigma = kernel_size / 3.0
        kernel_1d = torch.arange(kernel_size, dtype=torch.float32)
        kernel_1d = kernel_1d - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        return kernel_2d
    
    def _soft_dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Soft morphological dilation (完全可微分版本)
        Args:
            mask: torch.Tensor, shape [B, C, H, W], range [0,1]
        Returns:
            dilated_mask: torch.Tensor, shape [B, C, H, W], range [0,1]
        """
        if not self.use_soft_dilation:
            return mask
            
        original_mask = mask
        current_mask = mask
        
        for _ in range(self.dilation_iterations):
            # 使用 unfold 获取邻域
            unfolded = F.unfold(
                current_mask, 
                kernel_size=self.dilation_kernel_size, 
                padding=self.dilation_kernel_size // 2
            )  # [B, C*k*k, H*W]
            
            # Reshape 为 [B, C, k*k, H*W]
            B, _, HW = unfolded.shape
            C = current_mask.shape[1]
            k_squared = self.dilation_kernel_size ** 2
            unfolded = unfolded.view(B, C, k_squared, HW)
            
            # Log-sum-exp 操作（soft max）
            # 为了数值稳定性，先减去最大值
            max_vals, _ = torch.max(unfolded, dim=2, keepdim=True)
            exp_vals = torch.exp(self.dilation_beta * (unfolded - max_vals))
            sum_exp = torch.sum(exp_vals, dim=2, keepdim=True)
            
            # Log-sum-exp 公式
            lse = max_vals + torch.log(sum_exp) / self.dilation_beta
            lse = lse.squeeze(2)  # [B, C, H*W]
            
            # Reshape 回 [B, C, H, W]
            H, W = current_mask.shape[-2:]
            current_mask = lse.view(B, C, H, W)
            
            # 数值裁剪
            current_mask = torch.clamp(current_mask, 0, 1)
        
        # 可选：通过 dilation_strength 控制膨胀强度
        if self.dilation_strength != 1.0:
            # 线性插值：original * (1-strength) + dilated * strength
            current_mask = (1 - self.dilation_strength) * original_mask + self.dilation_strength * current_mask
            current_mask = torch.clamp(current_mask, 0, 1)
        
        return current_mask
    
    def _apply_sigmoid_activation(self, x: torch.Tensor) -> torch.Tensor:
        """应用sigmoid激活函数"""
        if self.use_sigmoid:
            return torch.sigmoid(x)
        return x
    
    def _create_affine_grid(self, source_size: Tuple[int, int], target_size: Tuple[int, int], 
                           batch_size: int, device: torch.device) -> torch.Tensor:
        """创建仿射变换的采样网格（纯PyTorch实现，保持梯度）"""
        src_h, src_w = source_size
        tgt_h, tgt_w = target_size
        
        # 计算缩放矩阵
        scale_x = src_w / tgt_w
        scale_y = src_h / tgt_h
        
        # 创建仿射变换矩阵 (batch_size, 2, 3)
        theta = torch.tensor([
            [scale_x, 0, 0],
            [0, scale_y, 0]
        ], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 生成采样网格
        grid = F.affine_grid(theta, (batch_size, 1, tgt_h, tgt_w), align_corners=True)
        return grid
    
    def _resize_with_grid_sample(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """使用grid_sample进行高质量插值调整（保持梯度）"""
        if x.shape[-2:] == target_size:
            return x
        
        batch_size = x.shape[0]
        
        if self.use_grid_sample:
            # 使用grid_sample进行高质量插值
            target_h, target_w = target_size
            
            # 创建标准化的采样网格 [-1, 1]
            y_coords = torch.linspace(-1, 1, target_h, device=x.device)
            x_coords = torch.linspace(-1, 1, target_w, device=x.device)
            
            # 创建网格
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 2]
            
            # 使用grid_sample进行采样
            result = F.grid_sample(
                x, 
                grid, 
                mode='bicubic',  # 高质量三次插值
                padding_mode='border',  # 边界处理
                align_corners=True
            )
            return result
        else:
            # fallback到标准插值
            return F.interpolate(
                x,
                size=target_size,
                mode='bicubic',
                align_corners=True,
                antialias=True
            )
    
    def _apply_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """应用平滑滤波"""
        if self.smooth_kernel is None:
            return x
        
        # 确保输入和kernel在同一设备上
        if self.smooth_kernel.device != x.device:
            self.smooth_kernel = self.smooth_kernel.to(x.device)
        
        padding = self.smooth_kernel_size // 2
        return F.conv2d(x, self.smooth_kernel, padding=padding)
    
    def _apply_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """应用阈值化处理"""
        if self.apply_threshold:
            return (x > self.threshold_value).float()
        return x
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """验证输入tensor的格式"""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {x.dim()}D tensor with shape {x.shape}")
        
        if x.shape[1] != 1:
            warnings.warn(f"Expected single channel mask, got {x.shape[1]} channels. Using first channel only.")
    
    def _convert_point_coordinates(
        self, 
        point_coords: torch.Tensor, 
        source_size: Tuple[int, int]
    ) -> torch.Tensor:
        """点坐标转换（保持梯度）"""
        if point_coords is None:
            return None

        source_h, source_w = source_size
        target_h, target_w = self.target_size

        scale_x = target_w / source_w
        scale_y = target_h / source_h

        # 保持梯度
        converted_coords = point_coords.float()
        
        # 确保在正确设备上
        if converted_coords.device != self.device:
            converted_coords = converted_coords.to(self.device)
        
        # 使用tensor操作保持梯度
        scale_tensor = torch.tensor([scale_x, scale_y], 
                                  device=converted_coords.device,
                                  dtype=converted_coords.dtype)
        converted_coords = converted_coords * scale_tensor
        
        # 使用torch.clamp保持梯度
        min_coords = torch.tensor([0, 0], 
                                device=converted_coords.device,
                                dtype=converted_coords.dtype)
        max_coords = torch.tensor([target_w - 1, target_h - 1], 
                                device=converted_coords.device,
                                dtype=converted_coords.dtype)
        converted_coords = torch.clamp(converted_coords, min=min_coords, max=max_coords)

        return converted_coords
    
    def _convert_box_coordinates(
        self,
        box_coords: torch.Tensor,
        source_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Box坐标转换（保持梯度）"""
        if box_coords is None:
            return None
            
        source_h, source_w = source_size
        target_h, target_w = self.target_size
        
        scale_x = target_w / source_w
        scale_y = target_h / source_h
        
        # 保持梯度
        converted_boxes = box_coords.float()
        
        # 确保在正确设备上
        if converted_boxes.device != self.device:
            converted_boxes = converted_boxes.to(self.device)
        
        # Box格式：[x_min, y_min, x_max, y_max]
        # 分别缩放x和y坐标
        scale_tensor = torch.tensor([scale_x, scale_y, scale_x, scale_y],
                                  device=converted_boxes.device,
                                  dtype=converted_boxes.dtype)
        converted_boxes = converted_boxes * scale_tensor
        
        # 限制到有效范围
        min_coords = torch.tensor([0, 0, 1, 1],  # 确保box有最小尺寸
                                device=converted_boxes.device,
                                dtype=converted_boxes.dtype)
        max_coords = torch.tensor([target_w - 1, target_h - 1, target_w - 1, target_h - 1],
                                device=converted_boxes.device,
                                dtype=converted_boxes.dtype)
        converted_boxes = torch.clamp(converted_boxes, min=min_coords, max=max_coords)
        
        return converted_boxes
    
    def forward(self, detris_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播（完全可微分）
        新增：Soft morphological dilation 步骤
        """
        # 输入验证
        self._validate_input(detris_output)
        
        # 确保在正确的设备上
        if detris_output.device != self.device:
            x = detris_output.to(self.device)
        else:
            x = detris_output
        
        # 只使用第一个通道
        if x.shape[1] > 1:
            x = x[:, :1, :, :]
        
        # 处理流水线（全部可微分）
        x = self._apply_sigmoid_activation(x)
        x = self._resize_with_grid_sample(x, self.target_size)
        x = self._apply_smoothing(x)
        
        # 新增：Soft morphological dilation
        x = self._soft_dilate_mask(x)
        
        x = self._apply_threshold(x)
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
    def forward_with_points(
        self, 
        detris_output: torch.Tensor, 
        point_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """同时处理mask和point的格式转换（完全可微分）"""
        # 获取源尺寸
        source_size = detris_output.shape[-2:]
        
        # 处理mask
        sam_mask_input = self.forward(detris_output)
        
        # 处理点坐标
        sam_point_coords = None
        if point_coords is not None:
            sam_point_coords = self._convert_point_coordinates(point_coords, source_size)
        
        return sam_mask_input, sam_point_coords
    
    def forward_with_boxes(
        self,
        detris_output: torch.Tensor,
        box_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """同时处理mask和box的格式转换（完全可微分）"""
        # 获取源尺寸
        source_size = detris_output.shape[-2:]
        
        # 处理mask
        sam_mask_input = self.forward(detris_output)
        
        # 处理box坐标转换
        sam_box_coords = None
        if box_coords is not None:
            sam_box_coords = self._convert_box_coordinates(box_coords, source_size)
        
        return sam_mask_input, sam_box_coords
    
    def forward_with_prompts(
        self,
        detris_output: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        box_coords: Optional[torch.Tensor] = None,
        generate_boxes_from_mask: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        统一处理所有类型的prompt（mask、point、box）
        
        Args:
            detris_output: DETRIS输出的mask
            point_coords: 可选的点坐标
            box_coords: 可选的box坐标  
            generate_boxes_from_mask: 是否从mask生成box
            
        Returns:
            Dict包含：
            - 'mask_inputs': 处理后的mask
            - 'point_coords': 处理后的点坐标（如果有）
            - 'boxes': 处理后的box坐标（如果有）
        """
        # 获取源尺寸
        source_size = detris_output.shape[-2:]
        
        # 处理mask（必须）
        sam_mask_input = self.forward(detris_output)
        
        results = {
            'mask_inputs': sam_mask_input
        }
        
        # 处理点坐标转换
        if point_coords is not None:
            sam_point_coords = self._convert_point_coordinates(point_coords, source_size)
            results['point_coords'] = sam_point_coords
        
        # 处理box坐标转换
        if box_coords is not None:
            sam_box_coords = self._convert_box_coordinates(box_coords, source_size)
            results['boxes'] = sam_box_coords
        
        # 从mask生成box（如果启用且有box sampler）
        if generate_boxes_from_mask and self.box_sampler is not None:
            generated_boxes = self.box_sampler(sam_mask_input)
            results['boxes'] = generated_boxes
        elif generate_boxes_from_mask and self.box_sampler is None:
            print("Warning: 请求从mask生成box，但box_sampler未初始化")
        
        return results
    
    def set_box_sampling(self, enabled: bool):
        """启用/禁用box采样"""
        if enabled and not BOX_SAMPLER_AVAILABLE:
            print("Warning: Box sampling不可用，box_sampler模块未导入")
            return
            
        if enabled and self.box_sampler is None:
            print("Warning: Box sampling未初始化，请在创建bridge时设置use_box_sampling=True")
            return
            
        self.use_box_sampling = enabled
        print(f"Box sampling {'启用' if enabled else '禁用'}")
    
    def set_box_sampler_params(self, **kwargs):
        """动态调整box sampler参数"""
        if self.box_sampler is None:
            print("Warning: Box sampler未初始化")
            return
            
        # 调用box sampler的参数设置方法
        if hasattr(self.box_sampler, 'set_quantiles') and 'quantile_low' in kwargs:
            self.box_sampler.set_quantiles(
                kwargs.get('quantile_low', self.box_sampler.quantile_low),
                kwargs.get('quantile_high', self.box_sampler.quantile_high)
            )
        
        if hasattr(self.box_sampler, 'set_temperature') and 'temperature' in kwargs:
            self.box_sampler.set_temperature(kwargs['temperature'])
    
    def set_soft_dilation_params(
        self, 
        use_soft_dilation: Optional[bool] = None,
        kernel_size: Optional[int] = None,
        iterations: Optional[int] = None,
        beta: Optional[float] = None,
        strength: Optional[float] = None
    ):
        """动态调整soft dilation参数"""
        if use_soft_dilation is not None:
            self.use_soft_dilation = use_soft_dilation
        if kernel_size is not None:
            self.dilation_kernel_size = kernel_size
        if iterations is not None:
            self.dilation_iterations = iterations
        if beta is not None:
            self.dilation_beta = beta
        if strength is not None:
            self.dilation_strength = strength
        
        print(f"Soft dilation 参数更新:")
        print(f"  - 启用: {self.use_soft_dilation}")
        if self.use_soft_dilation:
            print(f"  - 核大小: {self.dilation_kernel_size}")
            print(f"  - 迭代次数: {self.dilation_iterations}")
            print(f"  - Beta参数: {self.dilation_beta}")
            print(f"  - 强度: {self.dilation_strength}")
    
    def check_gradient_flow(self, detris_output: torch.Tensor) -> bool:
        """检查梯度流是否正常"""
        # 确保输入需要梯度
        if not detris_output.requires_grad:
            detris_output = detris_output.requires_grad_(True)
        
        # 前向传播
        output = self.forward(detris_output)
        
        # 检查输出是否需要梯度
        if not output.requires_grad:
            print(" 梯度断开：输出不需要梯度")
            return False
        
        # 模拟反向传播
        try:
            loss = output.sum()
            loss.backward()
            
            # 检查输入是否有梯度
            if detris_output.grad is None:
                print(" 梯度断开：输入没有梯度")
                return False
                
            print(" 梯度流正常 (包含soft dilation)")
            return True
            
        except Exception as e:
            print(f" 反向传播失败: {e}")
            return False
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'bridge_type': 'trainable',
            'target_size': self.target_size,
            'use_sigmoid': self.use_sigmoid,
            'use_grid_sample': self.use_grid_sample,
            'apply_threshold': self.apply_threshold,
            'threshold_value': self.threshold_value,
            'use_soft_dilation': self.use_soft_dilation,
            'dilation_kernel_size': self.dilation_kernel_size,
            'dilation_iterations': self.dilation_iterations,
            'dilation_beta': self.dilation_beta,
            'dilation_strength': self.dilation_strength,
            'use_box_sampling': self.use_box_sampling,
            'device': str(self.device)
        }
        
        # 添加box sampler配置
        if self.box_sampler is not None:
            config['box_sampler_config'] = self.box_sampler.get_config()
        
        return config


class AdaptiveBridge(nn.Module):
    """
    自适应桥梁：训练时保证梯度流，推理时使用高精度方法
    """
    def __init__(self, target_size: Tuple[int, int] = (256, 256), **kwargs):
        super().__init__()
        
        self.target_size = target_size
        
        # 训练时使用的bridge（完全可微分）
        self.trainable_bridge = TrainableBridge(target_size=target_size, **kwargs)
        
        # 推理时使用的bridge（高精度，可能断开梯度）
        try:
            # 尝试导入之前的高精度版本
            import cv2
            from .bridge import DetrisSamBridge
            self.inference_bridge = DetrisSamBridge(target_size=target_size, use_affine_transform=True)
            self.has_inference_bridge = True
        except:
            # 如果导入失败，使用同一个bridge
            self.inference_bridge = self.trainable_bridge
            self.has_inference_bridge = False
    
    def forward(self, detris_output: torch.Tensor) -> torch.Tensor:
        """根据训练/推理模式选择不同的bridge"""
        if self.training:
            # 训练模式：使用可微分版本
            return self.trainable_bridge(detris_output)
        else:
            # 推理模式：使用高精度版本
            return self.inference_bridge(detris_output)
    
    def forward_with_points(
        self, 
        detris_output: torch.Tensor, 
        point_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """根据训练/推理模式选择不同的bridge"""
        if self.training:
            return self.trainable_bridge.forward_with_points(detris_output, point_coords)
        else:
            return self.inference_bridge.forward_with_points(detris_output, point_coords)
    
    def forward_with_boxes(
        self,
        detris_output: torch.Tensor,
        box_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Box处理（根据模式选择）"""
        if self.training:
            return self.trainable_bridge.forward_with_boxes(detris_output, box_coords)
        else:
            return self.inference_bridge.forward_with_boxes(detris_output, box_coords)
    
    def forward_with_prompts(self, detris_output: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """统一prompt处理（根据模式选择）"""
        if self.training:
            return self.trainable_bridge.forward_with_prompts(detris_output, **kwargs)
        else:
            return self.inference_bridge.forward_with_prompts(detris_output, **kwargs)
    
    def check_gradient_flow(self, detris_output: torch.Tensor) -> bool:
        """检查当前模式下的梯度流"""
        if self.training:
            return self.trainable_bridge.check_gradient_flow(detris_output)
        else:
            print("  推理模式下不检查梯度流")
            return True
    
    def get_config(self) -> dict:
        """获取配置"""
        return {
            'bridge_type': 'adaptive',
            'target_size': self.target_size,
            'has_inference_bridge': self.has_inference_bridge,
            'training_mode': self.training
        }


class SimpleBridge(nn.Module):
    """
    简化版的桥梁，支持mask、point、box的基本转换
    """
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        super().__init__()
        self.target_size = target_size
        self._device = None
    
    @property
    def device(self) -> torch.device:
        """获取当前设备"""
        if self._device is not None:
            return self._device
        return torch.device('cpu')
    
    def forward(self, detris_output: torch.Tensor) -> torch.Tensor:
        """简单的mask转换：sigmoid + resize"""
        x = detris_output.to(self.device) if self._device else detris_output
        
        if x.shape[1] > 1:
            x = x[:, :1, :, :]
        
        x = torch.sigmoid(x)
        
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(
                x, 
                size=self.target_size, 
                mode='bicubic',
                align_corners=True,
                antialias=True
            )
        
        return x
    
    def forward_with_points(
        self, 
        detris_output: torch.Tensor, 
        point_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sam_mask_input = self.forward(detris_output)
        
        sam_point_coords = None
        if point_coords is not None:
            source_h, source_w = detris_output.shape[-2:]
            target_h, target_w = self.target_size
            
            scale_x = target_w / source_w
            scale_y = target_h / source_h
            
            sam_point_coords = point_coords.float()
            
            if sam_point_coords.device != sam_mask_input.device:
                sam_point_coords = sam_point_coords.to(sam_mask_input.device)
            
            scale_tensor = torch.tensor([scale_x, scale_y], 
                                      device=sam_point_coords.device,
                                      dtype=sam_point_coords.dtype)
            sam_point_coords = sam_point_coords * scale_tensor
            
            min_coords = torch.tensor([0, 0], 
                                    device=sam_point_coords.device,
                                    dtype=sam_point_coords.dtype)
            max_coords = torch.tensor([target_w - 1, target_h - 1], 
                                    device=sam_point_coords.device,
                                    dtype=sam_point_coords.dtype)
            sam_point_coords = torch.clamp(sam_point_coords, min=min_coords, max=max_coords)
        
        return sam_mask_input, sam_point_coords
    
    def forward_with_boxes(
        self,
        detris_output: torch.Tensor,
        box_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """简单的box转换"""
        sam_mask_input = self.forward(detris_output)
        
        sam_box_coords = None
        if box_coords is not None:
            source_h, source_w = detris_output.shape[-2:]
            target_h, target_w = self.target_size
            
            scale_x = target_w / source_w
            scale_y = target_h / source_h
            
            sam_box_coords = box_coords.float()
            
            if sam_box_coords.device != sam_mask_input.device:
                sam_box_coords = sam_box_coords.to(sam_mask_input.device)
            
            # Box格式：[x_min, y_min, x_max, y_max]
            scale_tensor = torch.tensor([scale_x, scale_y, scale_x, scale_y],
                                      device=sam_box_coords.device,
                                      dtype=sam_box_coords.dtype)
            sam_box_coords = sam_box_coords * scale_tensor
            
            # 限制到有效范围
            min_coords = torch.tensor([0, 0, 1, 1],
                                    device=sam_box_coords.device,
                                    dtype=sam_box_coords.dtype)
            max_coords = torch.tensor([target_w - 1, target_h - 1, target_w - 1, target_h - 1],
                                    device=sam_box_coords.device,
                                    dtype=sam_box_coords.dtype)
            sam_box_coords = torch.clamp(sam_box_coords, min=min_coords, max=max_coords)
        
        return sam_mask_input, sam_box_coords
    
    def forward_with_prompts(
        self,
        detris_output: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        box_coords: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """简单版的统一prompt处理"""
        sam_mask_input = self.forward(detris_output)
        
        results = {
            'mask_inputs': sam_mask_input
        }
        
        # 处理点坐标
        if point_coords is not None:
            _, sam_point_coords = self.forward_with_points(detris_output, point_coords)
            results['point_coords'] = sam_point_coords
        
        # 处理box坐标
        if box_coords is not None:
            _, sam_box_coords = self.forward_with_boxes(detris_output, box_coords)
            results['boxes'] = sam_box_coords
        
        return results
    
    def check_gradient_flow(self, detris_output: torch.Tensor) -> bool:
        """检查梯度流"""
        if not detris_output.requires_grad:
            detris_output = detris_output.requires_grad_(True)
        
        output = self.forward(detris_output)
        
        if not output.requires_grad:
            print(" 梯度断开：输出不需要梯度")
            return False
        
        try:
            loss = output.sum()
            loss.backward()
            
            if detris_output.grad is None:
                print(" 梯度断开：输入没有梯度")
                return False
                
            print(" 梯度流正常")
            return True
            
        except Exception as e:
            print(f" 反向传播失败: {e}")
            return False
    
    def get_config(self) -> dict:
        """获取配置"""
        return {
            'bridge_type': 'simple',
            'target_size': self.target_size,
            'device': str(self.device)
        }
    
    def to(self, device):
        """重写to方法"""
        result = super().to(device)
        result._device = device if isinstance(device, torch.device) else torch.device(device)
        return result


# 更新工厂函数以支持box sampling参数
def create_bridge(bridge_type: str = 'trainable', **kwargs) -> nn.Module:
    """
    工厂函数：创建不同类型的bridge
    
    Args:
        bridge_type: 'simple', 'trainable', 'adaptive'  
        **kwargs: bridge的配置参数
    
    Returns:
        bridge实例
    """
    # 定义各种bridge支持的参数
    SIMPLE_BRIDGE_PARAMS = {'target_size'}
    
    TRAINABLE_BRIDGE_PARAMS = {
        'target_size', 'use_sigmoid', 'apply_threshold', 
        'threshold_value', 'smooth_kernel_size', 'device', 'use_grid_sample',
        # soft dilation参数
        'use_soft_dilation', 'dilation_kernel_size', 'dilation_iterations', 
        'dilation_beta', 'dilation_strength',
        # 新增 box sampling 参数
        'use_box_sampling', 'box_sampler_type', 'box_sampler_config'
    }
    
    ADAPTIVE_BRIDGE_PARAMS = TRAINABLE_BRIDGE_PARAMS  # 相同参数
    
    # 过滤参数的辅助函数
    def filter_kwargs_for_bridge(supported_params, input_kwargs):
        filtered = {}
        ignored_params = []
        
        for key, value in input_kwargs.items():
            if key in supported_params:
                filtered[key] = value
            else:
                ignored_params.append(f"{key}={value}")
        
        if ignored_params:
            print(f"  Bridge忽略不支持的参数: {', '.join(ignored_params)}")
        
        return filtered
    
    # 根据bridge类型创建相应实例
    if bridge_type == 'simple':
        filtered_kwargs = filter_kwargs_for_bridge(SIMPLE_BRIDGE_PARAMS, kwargs)
        return SimpleBridge(**filtered_kwargs)
    
    elif bridge_type in ['trainable', 'default']:
        filtered_kwargs = filter_kwargs_for_bridge(TRAINABLE_BRIDGE_PARAMS, kwargs)
        return TrainableBridge(**filtered_kwargs)
    
    elif bridge_type == 'adaptive':
        filtered_kwargs = filter_kwargs_for_bridge(ADAPTIVE_BRIDGE_PARAMS, kwargs)
        return AdaptiveBridge(**filtered_kwargs)
    
    else:
        raise ValueError(f"Unknown bridge_type: {bridge_type}. Choose from 'simple', 'trainable', 'adaptive'")


# 便捷函数：创建支持box的bridge
def create_bridge_with_box_sampling(
    bridge_type: str = 'trainable',
    target_size: Tuple[int, int] = (256, 256),
    box_sampler_type: str = 'cdf',
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    **kwargs
) -> nn.Module:
    """
    便捷函数：创建启用box sampling的bridge
    
    Args:
        bridge_type: bridge类型
        target_size: 目标尺寸
        box_sampler_type: box sampler类型 ('cdf' 或 'adaptive')
        quantile_low: 下分位数
        quantile_high: 上分位数
        **kwargs: 其他bridge参数
    
    Returns:
        启用了box sampling的bridge实例
    """
    box_sampler_config = {
        'quantile_low': quantile_low,
        'quantile_high': quantile_high
    }
    
    return create_bridge(
        bridge_type=bridge_type,
        target_size=target_size,
        use_box_sampling=True,
        box_sampler_type=box_sampler_type,
        box_sampler_config=box_sampler_config,
        **kwargs
    )
