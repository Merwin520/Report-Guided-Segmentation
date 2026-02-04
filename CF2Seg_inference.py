#!/usr/bin/env python3
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import json
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

import utils.config as config
from utils.dataset import CXRDataset, tokenize
from model import build_segmenter

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='DETRIS Inference and Visualization (Fixed)')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--data-root', help='Dataset root directory')
    parser.add_argument('--mode', choices=['single', 'batch', 'dataset', 'debug'], default='dataset',
                        help='Inference mode (debug mode for testing)')
    
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--text', help='Text description for single image')
    parser.add_argument('--image-dir', help='Directory containing images')
    parser.add_argument('--text-file', help='Text file with descriptions')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val',
                        help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to process (-1 for all)')
    parser.add_argument('--output-dir', default='./inference_results_fixed',
                        help='Output directory')
    parser.add_argument('--vis-mode', choices=['overlay', 'side-by-side', 'grid', 'detailed', 'debug'], 
                        default='debug', help='Visualization mode')
    parser.add_argument('--save-raw', action='store_true', default=True,
                        help='Save raw prediction masks')
    parser.add_argument('--save-overlay', action='store_true', default=True,
                        help='Save overlay visualizations')
    parser.add_argument('--save-comparison', action='store_true', default=True,
                        help='Save comparison grids')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Segmentation threshold')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Overlay transparency')
    parser.add_argument('--colormap', default='red', choices=['red', 'blue', 'green', 'rainbow'],
                        help='Color scheme for masks')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 12],
                        help='Figure size for visualizations')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output image DPI')
    parser.add_argument('--debug-first-sample', action='store_true',
                        help='Debug the first sample in detail')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def load_config_and_model(config_path, checkpoint_path, device='cuda:0'):
    logger.info("Loading configuration and model...")
    cfg = config.load_cfg_from_cfg_file(config_path)
    flat_config = {}
    for section in ['DATA', 'TRAIN', 'COCOOP', 'CONTRASTIVE', 'LOSS', 'TEST', 'MISC']:
        if hasattr(cfg, section):
            section_cfg = getattr(cfg, section)
            if hasattr(section_cfg, 'items'):
                for key, value in section_cfg.items():
                    flat_config[key] = value
    
    for key, value in flat_config.items():
        setattr(cfg, key, value)
    model, _ = build_segmenter(cfg)
    model = model.to(device)
    model.eval()
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
        logger.info(f"Checkpoint IoU: {checkpoint.get('best_iou', 'Unknown')}")
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  
        else:
            new_state_dict[k] = v
    
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {}
    unexpected_keys = []
    coord_gate_keys = []
    
    for k, v in new_state_dict.items():
        if k in model_keys:
            filtered_state_dict[k] = v
        else:
            unexpected_keys.append(k)
            if 'coord_gate' in k:
                coord_gate_keys.append(k)
    missing_keys = []
    for k in model_keys:
        if k not in new_state_dict:
            missing_keys.append(k)
    
    if coord_gate_keys:
        logger.info(f"Found {len(coord_gate_keys)} coord_gate related keys in checkpoint (will be ignored due to dynamic initialization)")
        logger.debug(f"Coord_gate keys: {coord_gate_keys}")
    
    if unexpected_keys:
        non_coord_unexpected = [k for k in unexpected_keys if 'coord_gate' not in k]
        if non_coord_unexpected:
            logger.warning(f"Other unexpected keys: {non_coord_unexpected}")
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
    missing_keys_result, unexpected_keys_result = model.load_state_dict(filtered_state_dict, strict=False)
    
    logger.info("Model loaded successfully!")
    logger.info(f"Loaded {len(filtered_state_dict)} parameters, ignored {len(unexpected_keys)} mismatched keys")
    
    return model, cfg


def predict_sample(model, img_tensor, text, cfg, device, return_raw=False):
    model.eval()
    img = img_tensor.unsqueeze(0).to(device) 
    text_tensor = tokenize(text, cfg.word_len).to(device)  

    pred = model(img, text_tensor)  
    logger.debug(f"Model output shape: {pred.shape}, dtype: {pred.dtype}")
    logger.debug(f"Output value range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    pred_raw = pred.clone() if return_raw else None
    pred = torch.sigmoid(pred)
    logger.debug(f"After sigmoid range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    original_shape = pred.shape
    if len(pred.shape) == 4:  
        pred = pred.squeeze(0)  
        if pred.shape[0] == 1:  
            pred = pred.squeeze(0)  
        else:
            logger.debug(f"Multiple channels detected: {pred.shape[0]}, taking first channel")
            pred = pred[0]  # -> [H, W]
    elif len(pred.shape) == 3:  # [B, H, W] 或 [C, H, W]
        if pred.shape[0] == 1:  
            pred = pred.squeeze(0)  # -> [H, W]
        else:
            logger.debug(f"Ambiguous 3D shape: {pred.shape}, taking first dimension")
            pred = pred[0]  # -> [H, W]
    elif len(pred.shape) == 2:  # [H, W] 
        pass
    else:
        raise ValueError(f"Cannot handle prediction shape: {pred.shape}")
    
    logger.debug(f"Prediction shape transformation: {original_shape} -> {pred.shape}")
    

    input_size = img.shape[-2:]  # (H, W)
    pred_size = pred.shape
    
    if pred_size != input_size:
        logger.debug(f"Resizing prediction from {pred_size} to {input_size}")
        pred = pred.unsqueeze(0).unsqueeze(0)  
        pred = F.interpolate(
            pred,
            size=input_size,
            mode='bicubic',
            align_corners=True,
            antialias=True
        ).squeeze()  
    
    pred = torch.clamp(pred, 0.0, 1.0)
    

    if len(pred.shape) != 2:
        logger.error(f"Final prediction shape is not 2D: {pred.shape}")
        if pred.numel() == input_size[0] * input_size[1]:
            pred = pred.view(input_size[0], input_size[1])
        else:
            raise ValueError(f"Cannot reshape prediction to target size {input_size}")
    
    if return_raw and pred_raw is not None:
        if len(pred_raw.shape) == 4:
            pred_raw = pred_raw.squeeze(0)
            if pred_raw.shape[0] == 1:
                pred_raw = pred_raw.squeeze(0)
            else:
                pred_raw = pred_raw[0]
        elif len(pred_raw.shape) == 3:
            if pred_raw.shape[0] == 1:
                pred_raw = pred_raw.squeeze(0)
            else:
                pred_raw = pred_raw[0]
        
        return pred.cpu().numpy(), pred_raw.cpu().numpy()
    else:
        return pred.cpu().numpy()


def check_model_output_format(model, sample_img, sample_text, cfg, device):

    model.eval()
    
    with torch.no_grad():
        img = sample_img.unsqueeze(0).to(device)
        text_tensor = tokenize(sample_text, cfg.word_len).to(device)
        

        output = model(img, text_tensor)
        
        print("\n=== 模型输出格式检查 ===")
        print(f"输入图像形状: {img.shape}")
        print(f"输入文本形状: {text_tensor.shape}")
        print(f"模型输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 应用sigmoid后的检查
        output_sigmoid = torch.sigmoid(output)
        print(f"Sigmoid后值范围: [{output_sigmoid.min().item():.4f}, {output_sigmoid.max().item():.4f}]")
        print("=" * 40)
        
        return output


def inverse_transform_prediction(pred, mat_info, target_size):
    """将预测结果逆变换回原始尺寸 (改进版)"""
    w, h = target_size
    
    # 输入验证
    if pred is None or pred.size == 0:
        logger.error("Empty prediction received")
        return np.zeros((h, w), dtype=np.float32)
    
    # 确保pred是2D数组
    if len(pred.shape) != 2:
        logger.warning(f"Unexpected prediction shape: {pred.shape}, attempting to reshape")
        if pred.size == h * w:
            pred = pred.reshape(h, w)
        else:
            logger.error(f"Cannot reshape prediction of size {pred.size} to ({h}, {w})")
            return np.zeros((h, w), dtype=np.float32)
    
    try:
        # 验证和处理变换矩阵
        if mat_info is None:
            logger.debug("No transformation matrix provided, using direct resize")
            return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 确保矩阵类型和精度
        if isinstance(mat_info, (list, tuple)):
            mat = np.array(mat_info, dtype=np.float32)
        else:
            mat = mat_info.astype(np.float32)
        
        # 检查矩阵有效性
        if mat.shape != (2, 3):
            logger.warning(f"Invalid transformation matrix shape: {mat.shape}, expected (2, 3)")
            return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 检查矩阵是否包含异常值
        if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
            logger.warning("Transformation matrix contains NaN or Inf values")
            return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 检查行列式（避免奇异矩阵）
        det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        if abs(det) < 1e-6:
            logger.warning(f"Transformation matrix is nearly singular, det={det}")
            return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 使用仿射变换
        transformed = cv2.warpAffine(
            pred.astype(np.float32), 
            mat, 
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderValue=0.0
        )
        
        # 确保输出尺寸正确
        if transformed.shape != (h, w):
            logger.warning(f"Transformation output shape {transformed.shape} != target {(h, w)}")
            transformed = cv2.resize(transformed, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return transformed
        
    except Exception as e:
        logger.warning(f"Affine transform failed: {e}")
        logger.warning(f"Matrix info: {mat_info}")
        logger.warning(f"Pred shape: {pred.shape}, Target size: {target_size}")
        
        # 安全的fallback方案
        try:
            return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as e2:
            logger.error(f"Fallback resize also failed: {e2}")
            return np.zeros((h, w), dtype=np.float32)


def get_color_scheme(colormap, alpha=0.6):
    """获取颜色方案"""
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'rainbow': None  # 特殊处理
    }
    return colors.get(colormap, (255, 0, 0))


def create_overlay_visualization(original_img, pred_mask, gt_mask=None, 
                               alpha=0.6, pred_color=(255, 0, 0), gt_color=(0, 255, 0)):
    """创建叠加可视化"""
    overlay = original_img.copy()
    
    # 预测掩码叠加
    if pred_mask is not None:
        pred_colored = np.zeros_like(original_img)
        pred_colored[pred_mask > 0] = pred_color
        overlay = cv2.addWeighted(overlay, 1-alpha, pred_colored, alpha, 0)
    
    # 真实掩码叠加（如果有）
    if gt_mask is not None:
        gt_colored = np.zeros_like(original_img)
        gt_colored[gt_mask > 0] = gt_color
        overlay = cv2.addWeighted(overlay, 1-alpha/2, gt_colored, alpha/2, 0)
    
    return overlay


def create_debug_visualization(original_img, pred_mask, gt_mask=None, text="", 
                             pred_raw=None, figsize=(20, 12), iou=None):
    """创建调试可视化，包含更多信息"""
    plt.style.use('default')
    
    # 根据是否有gt_mask决定子图数量
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
    
    # 1. 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. 原始预测值（如果有）
    if pred_raw is not None:
        im1 = axes[1].imshow(pred_raw, cmap='viridis')
        axes[1].set_title(f'Raw Prediction\n[{pred_raw.min():.3f}, {pred_raw.max():.3f}]', 
                         fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].axis('off')
        axes[1].set_title('No Raw Prediction', fontsize=12)
    
    # 3. 二值化预测
    im2 = axes[2].imshow(pred_mask, cmap='Reds', alpha=0.8)
    pred_title = 'Binary Prediction'
    if iou is not None:
        pred_title += f' (IoU: {iou:.3f})'
    axes[2].set_title(pred_title, fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # 4. 预测叠加图
    if len(axes) > 3:
        overlay = create_overlay_visualization(original_img, pred_mask, alpha=0.5)
        axes[3].imshow(overlay)
        axes[3].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        axes[3].axis('off')
    
    if gt_mask is not None and len(axes) > 4:
        # 5. 真实掩码
        axes[4].imshow(gt_mask, cmap='Greens', alpha=0.8)
        axes[4].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[4].axis('off')
        
        # 6. 真实掩码叠加
        if len(axes) > 5:
            gt_overlay = create_overlay_visualization(original_img, gt_mask, 
                                                    alpha=0.5, pred_color=(0, 255, 0))
            axes[5].imshow(gt_overlay)
            axes[5].set_title('GT Overlay', fontsize=14, fontweight='bold')
            axes[5].axis('off')
        
        # 7. 对比图
        if len(axes) > 6:
            comparison = np.zeros_like(original_img)
            comparison[pred_mask > 0] = [255, 0, 0]  # 红色：预测
            comparison[gt_mask > 0] = [0, 255, 0]    # 绿色：真实
            comparison[np.logical_and(pred_mask > 0, gt_mask > 0)] = [255, 255, 0]  # 黄色：重叠
            
            comp_overlay = cv2.addWeighted(original_img, 0.7, comparison, 0.3, 0)
            axes[6].imshow(comp_overlay)
            axes[6].set_title('Comparison\n(Red=Pred, Green=GT, Yellow=Overlap)', 
                             fontsize=12, fontweight='bold')
            axes[6].axis('off')
        
        # 8. 统计信息
        if len(axes) > 7:
            axes[7].axis('off')
            
            # 计算详细统计
            intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
            union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
            pred_area = (pred_mask > 0).sum()
            gt_area = (gt_mask > 0).sum()
            total_pixels = pred_mask.size
            
            iou_calc = intersection / (union + 1e-7)
            precision = intersection / (pred_area + 1e-7)
            recall = intersection / (gt_area + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            
            stats_text = f"""Statistics:
IoU: {iou_calc:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1: {f1:.4f}

Pixel Counts:
Predicted: {pred_area:,} ({pred_area/total_pixels*100:.1f}%)
Ground Truth: {gt_area:,} ({gt_area/total_pixels*100:.1f}%)
Intersection: {intersection:,}
Union: {union:,}"""
            
            axes[7].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    else:
        # 隐藏多余的子图
        for i in range(3 if gt_mask is None else 4, len(axes)):
            axes[i].axis('off')
    
    # 添加文本描述
    plt.suptitle(f'Debug Visualization - Text: "{text}"', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    return fig


def create_detailed_grid(original_img, pred_mask, gt_mask=None, text="", 
                        pred_iou=None, figsize=(15, 10), dpi=150):
    """创建详细的网格可视化"""
    # 设置matplotlib样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        axes = axes.flatten()
    
    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 预测掩码
    title = 'Prediction'
    if pred_iou is not None:
        title += f' (IoU: {pred_iou:.3f})'
    axes[1].imshow(pred_mask, cmap='Reds', alpha=0.8)
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 预测叠加图
    overlay = create_overlay_visualization(original_img, pred_mask, alpha=0.5)
    axes[2].imshow(overlay)
    axes[2].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    if gt_mask is not None:
        # 真实掩码
        axes[3].imshow(gt_mask, cmap='Greens', alpha=0.8)
        axes[3].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        # 真实掩码叠加图
        gt_overlay = create_overlay_visualization(original_img, gt_mask, 
                                                alpha=0.5, pred_color=(0, 255, 0))
        axes[4].imshow(gt_overlay)
        axes[4].set_title('Ground Truth Overlay', fontsize=14, fontweight='bold')
        axes[4].axis('off')
        
        # 对比图 (预测=红色，真实=绿色，重叠=黄色)
        comparison = np.zeros_like(original_img)
        comparison[pred_mask > 0] = [255, 0, 0]  # 红色：预测
        comparison[gt_mask > 0] = [0, 255, 0]    # 绿色：真实
        comparison[np.logical_and(pred_mask > 0, gt_mask > 0)] = [255, 255, 0]  # 黄色：重叠
        
        comp_overlay = cv2.addWeighted(original_img, 0.7, comparison, 0.3, 0)
        axes[5].imshow(comp_overlay)
        axes[5].set_title('Comparison (Red=Pred, Green=GT, Yellow=Overlap)', 
                         fontsize=12, fontweight='bold')
        axes[5].axis('off')
    else:
        # 隐藏多余的子图
        for i in range(3, len(axes)):
            axes[i].axis('off')
    
    # 添加文本描述
    plt.suptitle(f'Text: "{text}"', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    return fig


def create_side_by_side_visualization(original_img, pred_mask, gt_mask=None, text=""):
    """创建并排对比可视化"""
    h, w = original_img.shape[:2]
    
    if gt_mask is not None:
        # 原图 | 预测 | 真实掩码
        canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
        canvas[:, :w] = original_img
        canvas[:, w:2*w] = np.stack([pred_mask*255]*3, axis=-1)
        canvas[:, 2*w:] = np.stack([gt_mask*255]*3, axis=-1)
        
        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Prediction', (w+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Ground Truth', (2*w+10, 30), font, 1, (255, 255, 255), 2)
    else:
        # 原图 | 预测
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        canvas[:, :w] = original_img
        canvas[:, w:] = np.stack([pred_mask*255]*3, axis=-1)
        
        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Prediction', (w+10, 30), font, 1, (255, 255, 255), 2)
    
    return canvas


def validate_transformation_matrix(mat_info):
    """验证变换矩阵的有效性"""
    try:
        if isinstance(mat_info, (list, tuple)):
            mat = np.array(mat_info, dtype=np.float32)
        else:
            mat = mat_info.astype(np.float32)
        
        if mat.shape != (2, 3):
            return False, f"Invalid shape: {mat.shape}, expected (2, 3)"
        
        # 检查是否包含NaN或Inf
        if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
            return False, "Matrix contains NaN or Inf values"
        
        # 检查行列式是否接近0（会导致变换失败）
        det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        if abs(det) < 1e-6:
            return False, f"Transformation matrix is nearly singular, det={det}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Exception during validation: {e}"


def calculate_simple_iou(pred_mask, gt_mask):
    """计算简单IoU（用于可视化标注）"""
    if gt_mask is None:
        return None
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-7)
    return iou


def check_prediction_quality(pred_original, original_size, threshold=0.5):
    """检查预测质量和可能的问题"""
    h, w = original_size
    pred_h, pred_w = pred_original.shape
    
    issues = []
    
    # 检查尺寸匹配
    if (pred_h, pred_w) != (h, w):
        issues.append(f"Size mismatch: pred({pred_h}, {pred_w}) vs target({h}, {w})")
    
    # 检查预测值范围
    pred_min, pred_max = pred_original.min(), pred_original.max()
    if pred_min < 0 or pred_max > 1:
        issues.append(f"Prediction values out of range [0,1]: [{pred_min:.3f}, {pred_max:.3f}]")
    
    # 检查是否全零或全一
    if pred_max < 1e-6:
        issues.append("Prediction is nearly all zeros")
    elif pred_min > 1-1e-6:
        issues.append("Prediction is nearly all ones")
    
    # 检查二值化后的区域大小
    binary_mask = (pred_original > threshold).astype(np.uint8)
    positive_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    positive_ratio = positive_pixels / total_pixels
    
    if positive_ratio < 0.001:
        issues.append(f"Very small predicted region: {positive_ratio:.1%} of image")
    elif positive_ratio > 0.8:
        issues.append(f"Very large predicted region: {positive_ratio:.1%} of image")
    
    return issues


def debug_first_sample(args, model, cfg, device):
    """调试第一个样本的详细函数"""
    logger.info("Starting debug mode for first sample...")
    
    # 创建数据集
    dataset = CXRDataset(
        data_root=args.data_root or cfg.data_root,
        split=args.split,
        mode='test',
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        use_augmentation=False,
        aug_probability=0.0
    )
    
    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return
    
    # 获取第一个样本
    img_tensor, params = dataset[0]
    
    # 获取样本信息
    img_filename = params['img_filename']
    description = params['description']
    ori_size = params['ori_size']
    original_img = params['ori_img']
    
    logger.info(f"Debug sample: {img_filename}")
    logger.info(f"Description: {description}")
    logger.info(f"Original size: {ori_size}")
    logger.info(f"Tensor shape: {img_tensor.shape}")
    
    # 检查模型输出格式
    check_model_output_format(model, img_tensor, description, cfg, device)
    
    # 进行推理（包含原始预测）
    pred, pred_raw = predict_sample(model, img_tensor, description, cfg, device, return_raw=True)
    
    logger.info(f"Processed prediction shape: {pred.shape}")
    logger.info(f"Processed prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    
    # 转换回原始尺寸
    h, w = ori_size
    inverse_mat = params['inverse']
    pred_original = inverse_transform_prediction(pred, inverse_mat, (w, h))
    
    # 修复预测值范围
    pred_original = np.clip(pred_original, 0.0, 1.0)
    
    # 应用阈值
    pred_mask = (pred_original > args.threshold).astype(np.uint8)
    
    # 加载真实掩码
    gt_mask = None
    mask_path = params['mask_path']
    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = (gt_mask / 255.0).astype(np.uint8)
    
    # 计算IoU
    iou = calculate_simple_iou(pred_mask, gt_mask)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建调试可视化
    fig = create_debug_visualization(
        original_img, pred_mask, gt_mask, description, 
        pred_raw, args.figsize, iou
    )
    
    # 保存调试结果
    debug_path = os.path.join(args.output_dir, f"debug_{img_filename.split('.')[0]}.png")
    fig.savefig(debug_path, dpi=args.dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    logger.info(f"Debug visualization saved to: {debug_path}")
    if iou is not None:
        logger.info(f"IoU: {iou:.4f}")


def inference_single_image(args, model, cfg, device):
    """单张图片推理"""
    logger.info(f"Processing single image: {args.image}")
    
    # 读取和预处理图像（这里需要根据你的预处理函数调整）
    # 假设你有一个预处理函数
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # TODO: 添加你的预处理逻辑
    # img_tensor, mat_inv, ori_size = preprocess_image(args.image, cfg.input_size)
    
    logger.info("Single image inference not fully implemented - need preprocessing function")


def inference_dataset(args, model, cfg, device):
    """数据集推理"""
    logger.info(f"Processing dataset split: {args.split}")
    
    # 创建数据集
    dataset = CXRDataset(
        data_root=args.data_root or cfg.data_root,
        split=args.split,
        mode='test',  # 使用test模式
        input_size=cfg.input_size,
        word_length=cfg.word_len,
        use_augmentation=False,  # 测试时不使用增强
        aug_probability=0.0
    )
    
    # 确定处理样本数
    num_samples = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))
    logger.info(f"Processing {num_samples} samples from {len(dataset)} total samples")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建子目录
    if args.save_raw:
        os.makedirs(os.path.join(args.output_dir, 'raw_predictions'), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.output_dir, 'overlays'), exist_ok=True)
    if args.save_comparison:
        os.makedirs(os.path.join(args.output_dir, 'comparisons'), exist_ok=True)
    
    # 结果统计
    results = []
    
    for idx in tqdm(range(num_samples), desc='Processing samples'):
        try:
            # 获取样本
            img_tensor, params = dataset[idx]
            
            # 获取样本信息
            img_filename = params['img_filename']
            mask_filename = params['mask_filename']
            mask_path = params['mask_path']
            description = params['description']
            ori_size = params['ori_size']
            inverse_mat = params['inverse']
            original_img = params['ori_img']
            
            if args.verbose:
                logger.info(f"Processing {idx+1}/{num_samples}: {img_filename}")
            
            # 推理（可能包含原始预测用于调试）
            if args.vis_mode == 'debug':
                pred, pred_raw = predict_sample(model, img_tensor, description, cfg, device, return_raw=True)
            else:
                pred = predict_sample(model, img_tensor, description, cfg, device, return_raw=False)
                pred_raw = None
            
            # 转换回原始尺寸
            h, w = ori_size
            pred_original = inverse_transform_prediction(pred, inverse_mat, (w, h))
            
            # 修复预测值范围
            pred_original = np.clip(pred_original, 0.0, 1.0)
            
            # 检查预测质量
            quality_issues = check_prediction_quality(pred_original, (h, w), args.threshold)
            if quality_issues and args.verbose:
                logger.warning(f"Quality issues for {img_filename}: {'; '.join(quality_issues)}")
            
            # 应用阈值
            pred_mask = (pred_original > args.threshold).astype(np.uint8)
            
            # 加载真实掩码
            gt_mask = None
            iou = None
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask = (gt_mask / 255.0).astype(np.uint8)
                    iou = calculate_simple_iou(pred_mask, gt_mask)
            
            # 创建文件名前缀
            prefix = f"sample_{idx+1:03d}_{img_filename.split('.')[0]}"
            if iou is not None:
                prefix += f"_iou_{iou:.3f}"
            
            # 保存原始预测
            if args.save_raw:
                raw_path = os.path.join(args.output_dir, 'raw_predictions', f"{prefix}_raw.png")
                cv2.imwrite(raw_path, (pred_original * 255).astype(np.uint8))
                
                binary_path = os.path.join(args.output_dir, 'raw_predictions', f"{prefix}_binary.png")
                cv2.imwrite(binary_path, (pred_mask * 255).astype(np.uint8))
            
            # 创建可视化
            if args.vis_mode == 'overlay' and args.save_overlay:
                overlay = create_overlay_visualization(original_img, pred_mask, gt_mask, args.alpha)
                overlay_path = os.path.join(args.output_dir, 'overlays', f"{prefix}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            elif args.vis_mode == 'side-by-side' and args.save_overlay:
                side_by_side = create_side_by_side_visualization(original_img, pred_mask, gt_mask, description)
                side_path = os.path.join(args.output_dir, 'overlays', f"{prefix}_side_by_side.png")
                cv2.imwrite(side_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
            
            elif args.vis_mode == 'detailed' and args.save_comparison:
                fig = create_detailed_grid(original_img, pred_mask, gt_mask, 
                                         description, iou, args.figsize, args.dpi)
                comparison_path = os.path.join(args.output_dir, 'comparisons', f"{prefix}_detailed.png")
                fig.savefig(comparison_path, dpi=args.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
            
            elif args.vis_mode == 'debug' and args.save_comparison:
                fig = create_debug_visualization(
                    original_img, pred_mask, gt_mask, description, 
                    pred_raw, args.figsize, iou
                )
                debug_path = os.path.join(args.output_dir, 'comparisons', f"{prefix}_debug.png")
                fig.savefig(debug_path, dpi=args.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
            
            # 记录结果
            result = {
                'index': idx,
                'filename': img_filename,
                'description': description,
                'iou': iou,
                'has_gt': gt_mask is not None,
                'quality_issues': quality_issues
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            if args.verbose:
                import traceback
                logger.error(traceback.format_exc())
            continue
    
    # 保存结果摘要
    if results:
        results_file = os.path.join(args.output_dir, 'inference_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'config': args.config,
                'checkpoint': args.checkpoint,
                'split': args.split,
                'num_samples': num_samples,
                'visualization_mode': args.vis_mode,
                'threshold': args.threshold,
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # 打印统计信息
        valid_ious = [r['iou'] for r in results if r['iou'] is not None]
        if valid_ious:
            mean_iou = np.mean(valid_ious)
            logger.info(f"Mean IoU: {mean_iou:.4f} ({len(valid_ious)} samples with GT)")
        
        # 统计质量问题
        samples_with_issues = [r for r in results if r['quality_issues']]
        if samples_with_issues:
            logger.warning(f"Found quality issues in {len(samples_with_issues)} samples")
    
    logger.info("Dataset inference completed!")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level=log_level)
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model, cfg = load_config_and_model(args.config, args.checkpoint, device)
    
    # 执行推理
    if args.mode == 'debug' or args.debug_first_sample:
        debug_first_sample(args, model, cfg, device)
    
    elif args.mode == 'single':
        if not args.image or not args.text:
            logger.error("Single mode requires --image and --text arguments")
            return
        inference_single_image(args, model, cfg, device)
    
    elif args.mode == 'dataset':
        if not args.data_root and not hasattr(cfg, 'data_root'):
            logger.error("Dataset mode requires --data-root argument or data_root in config")
            return
        inference_dataset(args, model, cfg, device)
    
    else:
        logger.error(f"Mode {args.mode} not implemented yet")


if __name__ == '__main__':
    main()
