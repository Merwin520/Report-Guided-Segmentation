"""
集成模型训练引擎（修复版）
修复指标计算的double sigmoid问题和验证逻辑
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Tuple, Optional
import cv2
import os
from utils.integrated_loss import compute_integrated_loss, calculate_batch_metrics


def calculate_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """计算分割指标（调用 utils.integrated_loss）"""
    from utils.integrated_loss import calculate_metrics as calc_metrics
    return calc_metrics(pred_mask, gt_mask, threshold, input_is_logits=False)  # 修复：明确指定是概率


def convert_batch_to_sam_format(sam_batch_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """将批次字典转换为 SAM 输入格式"""
    batch_size = sam_batch_dict['image'].shape[0]
    sam_batched_input = []

    for i in range(batch_size):
        sample_dict = {'image': sam_batch_dict['image'][i].float()}

        if 'original_size' in sam_batch_dict:
            size_tensor = sam_batch_dict['original_size'][0] if isinstance(sam_batch_dict['original_size'], list) else sam_batch_dict['original_size']
            sample_dict['original_size'] = (int(size_tensor[i*2].item()), int(size_tensor[i*2+1].item())) if len(size_tensor) > i*2+1 else sam_batch_dict['image'][i].shape[-2:]
        else:
            sample_dict['original_size'] = sam_batch_dict['image'][i].shape[-2:]

        if 'point_coords' in sam_batch_dict and sam_batch_dict['point_coords'] is not None:
            sample_dict['point_coords'] = sam_batch_dict['point_coords'][i].float()
        if 'point_labels' in sam_batch_dict and sam_batch_dict['point_labels'] is not None:
            sample_dict['point_labels'] = sam_batch_dict['point_labels'][i].float()
        if 'image_filename' in sam_batch_dict:
            sample_dict['image_filename'] = sam_batch_dict['image_filename'][i] if i < len(sam_batch_dict['image_filename']) else None

        sam_batched_input.append(sample_dict)

    return sam_batched_input


def process_sam_output_unified(sam_output: List[Dict[str, torch.Tensor]], 
                              original_sizes: List[tuple], 
                              target_size: tuple,
                              multimask_output: bool = True,
                              return_logits: bool = True) -> torch.Tensor:
    """
    统一的SAM输出处理函数
    """
    batch_results = []

    for i, output in enumerate(sam_output):
        # 使用low_res_logits保持梯度流
        if return_logits and 'low_res_logits' in output:
            raw_output = output['low_res_logits']
        elif not return_logits and 'masks' in output:
            raw_output = output['masks']
        else:
            raw_output = output.get('low_res_logits', output.get('masks'))
        
        scores = output['iou_predictions']

        # 更安全的维度处理
        if raw_output.dim() == 4 and raw_output.shape[0] == 1:
            raw_output = raw_output.squeeze(0)
        if scores.dim() == 2 and scores.shape[0] == 1:
            scores = scores.squeeze(0)

        # 多mask选择
        if multimask_output and raw_output.shape[0] > 1:
            best_idx = scores.argmax()
            best_output = raw_output[best_idx]
        else:
            best_output = raw_output[0] if raw_output.dim() > 2 else raw_output

        # 确保是2D
        while best_output.dim() > 2:
            best_output = best_output.squeeze(0)

        # 插值到目标尺寸
        if best_output.shape != target_size:
            best_output_4d = best_output.unsqueeze(0).unsqueeze(0).float()
            best_output_resized = F.interpolate(
                best_output_4d, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            best_output = best_output_resized.squeeze(0).squeeze(0)

        # 类型转换
        if 'low_res_logits' in output and return_logits:
            pass  # 已经是logits
        elif 'masks' in output and not return_logits:
            pass  # 已经是概率
        elif 'low_res_logits' in output and not return_logits:
            best_output = torch.sigmoid(best_output)
        elif 'masks' in output and return_logits:
            logger.warning("Converting masks to logits will lose gradients!")
            best_output = torch.clamp(best_output, 1e-7, 1-1e-7)
            best_output = torch.log(best_output / (1 - best_output))

        best_output = best_output.float()
        batch_results.append(best_output)

    result = torch.stack(batch_results, dim=0)
    return result


def prepare_batch_data(batch_data: Dict[str, Any], model_dtype: torch.dtype) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, List[str]]:
    """
    统一的批次数据准备函数
    
    Returns:
        sam_batched_input: SAM格式的输入
        gt_mask: 真实标签mask (如果存在)
        text_prompts: 文本提示
    """
    sam_batch_dict = batch_data['sam_batched_input']
    text_prompts = batch_data['text_prompt']
    
    # 检查是否有GT
    gt_mask = None
    if 'mask_sam' in batch_data:
        gt_mask = batch_data['mask_sam'].cuda(non_blocking=True).to(model_dtype)
    
    # 转换输入格式
    sam_batched_input = convert_batch_to_sam_format(sam_batch_dict)
    for i in range(len(sam_batched_input)):
        sam_batched_input[i]['image'] = sam_batched_input[i]['image'].cuda(non_blocking=True).to(model_dtype)
        if 'point_coords' in sam_batched_input[i] and sam_batched_input[i]['point_coords'] is not None:
            sam_batched_input[i]['point_coords'] = sam_batched_input[i]['point_coords'].cuda(non_blocking=True).to(model_dtype)
        if 'point_labels' in sam_batched_input[i] and sam_batched_input[i]['point_labels'] is not None:
            sam_batched_input[i]['point_labels'] = sam_batched_input[i]['point_labels'].cuda(non_blocking=True).to(model_dtype)
    
    return sam_batched_input, gt_mask, text_prompts


def forward_pass(model, sam_batched_input: List[Dict[str, torch.Tensor]], 
                text_prompts: List[str], gt_mask: torch.Tensor, 
                cfg) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    统一的前向传播函数
    
    Returns:
        loss: 计算的损失
        pred_logits: 预测的logits
        batch_metrics: 批次指标
    """
    # 模型前向传播
    sam_output = model(batched_input=sam_batched_input, multimask_output=True, text_prompts=text_prompts)
    
    # 获取原始尺寸和目标尺寸
    original_sizes = [sam_batched_input[i]['original_size'] for i in range(len(sam_batched_input))]
    target_size = gt_mask.shape[-2:]
    
    # 处理SAM输出
    pred_logits = process_sam_output_unified(
        sam_output, 
        original_sizes, 
        target_size=target_size,
        multimask_output=True, 
        return_logits=True
    )
    
    # 计算损失
    loss = compute_integrated_loss(pred_logits, gt_mask, cfg)
    
    # 修复：计算指标（用于监控，不影响梯度）
    with torch.no_grad():
        pred_probs = torch.sigmoid(pred_logits.detach())
        batch_metrics = calculate_batch_metrics(pred_probs, gt_mask, input_is_logits=False, enable_hd95=False)  # 修复：明确指定是概率
    
    return loss, pred_logits, batch_metrics


def find_optimal_threshold(val_loader, model, cfg, num_batches: int = 15) -> float:
    """
    在验证开始时寻找最优阈值
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        cfg: 配置
        num_batches: 用于搜索的批次数量
        
    Returns:
        optimal_threshold: 最优阈值
    """
    model.eval()
    model_dtype = next(model.parameters()).dtype
    
    threshold_candidates = getattr(cfg, 'threshold_candidates', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    threshold_scores = {t: [] for t in threshold_candidates}
    
    # 修复1：添加日志控制变量，避免重复输出
    should_log = (not hasattr(cfg, 'rank') or cfg.rank == 0)
    
    with torch.no_grad():
        batch_count = 0
        for batch_data in val_loader:
            if batch_count >= num_batches:
                break
                
            try:
                sam_batched_input, gt_mask, text_prompts = prepare_batch_data(batch_data, model_dtype)
                
                if gt_mask is None:
                    continue
                    
                # 前向传播
                sam_output = model(batched_input=sam_batched_input, multimask_output=True, text_prompts=text_prompts)
                original_sizes = [sam_batched_input[i]['original_size'] for i in range(len(sam_batched_input))]
                target_size = gt_mask.shape[-2:]
                
                pred_logits = process_sam_output_unified(
                    sam_output, original_sizes, target_size=target_size,
                    multimask_output=True, return_logits=True
                )
                
                pred_probs = torch.sigmoid(pred_logits)
                
                # 修复2：测试不同阈值，使用概率而不是二值化结果
                for threshold in threshold_candidates:
                    metrics = calculate_batch_metrics(pred_probs, gt_mask, threshold=threshold, input_is_logits=False)
                    threshold_scores[threshold].append(metrics['dice'])
                
                batch_count += 1
                
                # 清理显存
                del sam_output, pred_logits, pred_probs
                if batch_count % 3 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if should_log:
                    logger.warning(f"Error in threshold search batch {batch_count}: {e}")
                continue
    
    best_threshold = 0.5  # 默认阈值
    best_avg_dice = 0.0
    valid_results = []
    
    for threshold, scores in threshold_scores.items():
        if scores:
            avg_dice = np.mean(scores)
            valid_results.append((threshold, avg_dice, len(scores)))
            if avg_dice > best_avg_dice:
                best_avg_dice = avg_dice
                best_threshold = threshold
    
    if should_log:
        if valid_results:
            # 排序显示前3个最佳结果，但只输出最终选择的一个
            valid_results.sort(key=lambda x: x[1], reverse=True)
            top_result = valid_results[0]
            logger.info(f"Optimal threshold found: {best_threshold:.2f} (Avg Dice: {best_avg_dice:.4f})")
        else:
            logger.warning(f"No valid threshold results found, using default: {best_threshold:.2f}")
    
    return best_threshold


def train_integrated(train_loader, model, optimizer, scheduler, scaler, epoch, cfg):
    """集成模型训练"""
    model.train()
    model_dtype = next(model.parameters()).dtype

    metrics_tracker = {'loss': [], 'iou': [], 'dice': [], 'lr': []}
    print_freq = getattr(cfg, 'print_freq', 50)
    empty_cache_freq = getattr(cfg, 'empty_cache_freq', 5)

    start_time = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        try:
            # 统一的数据准备
            sam_batched_input, gt_mask, text_prompts = prepare_batch_data(batch_data, model_dtype)

            optimizer.zero_grad()
            
            # 统一的前向传播（已修复指标计算）
            loss, pred_logits, batch_metrics = forward_pass(
                model, sam_batched_input, text_prompts, gt_mask, cfg
            )
            
            # 反向传播
            loss.backward()

            # 梯度裁剪
            max_norm = getattr(cfg, 'max_norm', 0.0)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            # 记录指标
            metrics_tracker['loss'].append(loss.item())
            metrics_tracker['iou'].append(batch_metrics['iou'])
            metrics_tracker['dice'].append(batch_metrics['dice'])
            metrics_tracker['lr'].append(optimizer.param_groups[0]['lr'])

            # 显存管理
            del loss, pred_logits
            if batch_idx % empty_cache_freq == 0:
                torch.cuda.empty_cache()

            # 打印进度
            if batch_idx % print_freq == 0:
                if len(metrics_tracker['loss']) > 0:
                    avg_loss = np.mean(metrics_tracker['loss'][-print_freq:])
                    avg_iou = np.mean(metrics_tracker['iou'][-print_freq:])
                    avg_dice = np.mean(metrics_tracker['dice'][-print_freq:])
                    lr = metrics_tracker['lr'][-1]
                    logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {avg_loss:.4f} IoU: {avg_iou:.4f} Dice: {avg_dice:.4f} LR: {lr:.2e}")
                else:
                    logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - Processing first batch...")

        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            raise

    # 计算epoch指标
    if len(metrics_tracker['loss']) > 0:
        epoch_metrics = {
            'loss': np.mean(metrics_tracker['loss']),
            'iou': np.mean(metrics_tracker['iou']),
            'dice': np.mean(metrics_tracker['dice']),
            'lr': metrics_tracker['lr'][-1],
            'time': time.time() - start_time
        }
        logger.info(f"Epoch {epoch} Training - Loss: {epoch_metrics['loss']:.4f} IoU: {epoch_metrics['iou']:.4f} Dice: {epoch_metrics['dice']:.4f} Time: {epoch_metrics['time']:.1f}s")
    else:
        epoch_metrics = {
            'loss': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'lr': 0.0,
            'time': time.time() - start_time
        }
        logger.error(f"Epoch {epoch} Training failed - No metrics recorded. Time: {epoch_metrics['time']:.1f}s")
    
    return epoch_metrics



def validate_integrated(val_loader, model, epoch, cfg):
    """集成模型验证 - 修复重复日志和阈值不一致问题"""
    model.eval()  # 设置为评估模式
    model_dtype = next(model.parameters()).dtype

    metrics_tracker = {'loss': [], 'iou': [], 'dice': [], 'precision': [], 'recall': [], 'f1': []}
    save_visualizations = getattr(cfg, 'visualize', False) and (epoch % 10 == 0)
    
    # 统一阈值选择逻辑，避免重复调用
    use_dynamic_threshold = getattr(cfg, 'use_dynamic_threshold', True)
    optimal_threshold = None
    
    if use_dynamic_threshold:
        # 保持原有接口，但内部避免重复日志
        optimal_threshold = find_optimal_threshold(val_loader, model, cfg, num_batches=len(val_loader))
    else:
        optimal_threshold = getattr(cfg, 'validation_threshold', 0.5)
    
    vis_save_dir = os.path.join(cfg.output_dir, f'visualizations_epoch_{epoch}') if save_visualizations else None
    if vis_save_dir and hasattr(cfg, 'rank') and cfg.rank == 0:
        os.makedirs(vis_save_dir, exist_ok=True)

    start_time = time.time()

    # 验证时使用 torch.no_grad() 节省显存
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            try:
                # 统一的数据准备
                sam_batched_input, gt_mask, text_prompts = prepare_batch_data(batch_data, model_dtype)
                
                # 检查是否有GT
                has_gt = gt_mask is not None
                if not has_gt:
                    continue

                # 模型推理 - 在no_grad下进行，节省显存
                sam_output = model(batched_input=sam_batched_input, multimask_output=True, text_prompts=text_prompts)

                original_sizes = [sam_batched_input[i]['original_size'] for i in range(len(sam_batched_input))]
                target_size = gt_mask.shape[-2:]

                # 处理SAM输出
                pred_logits = process_sam_output_unified(
                    sam_output, 
                    original_sizes, 
                    target_size=target_size,
                    multimask_output=True, 
                    return_logits=True
                )
                
                # 计算损失和指标
                loss = compute_integrated_loss(pred_logits, gt_mask, cfg)
                pred_probs = torch.sigmoid(pred_logits)
                
                # 修复2：使用统一的最优阈值计算指标
                batch_metrics = calculate_batch_metrics(
                    pred_probs, gt_mask, 
                    threshold=optimal_threshold, 
                    input_is_logits=False, 
                    enable_hd95=False
                )
                
                # 记录指标
                metrics_tracker['loss'].append(loss.item())
                for key in ['iou', 'dice', 'precision', 'recall', 'f1']:
                    metrics_tracker[key].append(batch_metrics[key])

                # 显存管理
                del sam_output, pred_logits, pred_probs
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                raise

    # 计算epoch平均指标
    epoch_metrics = {'time': time.time() - start_time}
    for key in metrics_tracker:
        if metrics_tracker[key]:
            epoch_metrics[key] = np.mean(metrics_tracker[key])
        else:
            epoch_metrics[key] = None

    # 统一日志输出，避免重复记录
    # 只在主进程且有有效指标时输出
    if epoch_metrics['iou'] is not None and (not hasattr(cfg, 'rank') or cfg.rank == 0):
        logger.info(
            f"Epoch {epoch} Validation - "
            f"Loss: {epoch_metrics['loss']:.4f} "
            f"IoU: {epoch_metrics['iou']:.4f} "
            f"Dice: {epoch_metrics['dice']:.4f} "
            f"Precision: {epoch_metrics['precision']:.4f} "
            f"Recall: {epoch_metrics['recall']:.4f} "
            f"F1: {epoch_metrics['f1']:.4f} "
            f"Threshold: {optimal_threshold:.2f} "
            f"Time: {epoch_metrics['time']:.1f}s"
        )
    elif epoch_metrics['iou'] is None and (not hasattr(cfg, 'rank') or cfg.rank == 0):
        logger.info(f"Epoch {epoch} Validation completed in {epoch_metrics['time']:.1f}s (no GT available)")

    return epoch_metrics
    

def test_integrated(test_loader, model, cfg, enable_hd95: bool = True):
    """
    集成模型测试函数 - 针对验证集合
    输出详细的评估指标，包括PA和HD95
    
    Args:
        test_loader: 测试数据加载器
        model: 训练好的模型
        cfg: 配置对象
        compute_hd95: 是否计算HD95（计算量较大，默认True）
        
    Returns:
        test_results: 包含所有指标的结果字典
    """
    model.eval()  # 设置为评估模式
    model_dtype = next(model.parameters()).dtype
    
    # 指标收集器
    all_metrics = {
        'iou': [], 'dice': [], 'precision': [], 'recall': [], 
        'f1': [], 'pa': [], 'hd95': []
    }
    
    # 测试配置
    test_threshold = getattr(cfg, 'test_threshold', 0.5)
    
    logger.info(f"开始测试，使用阈值: {test_threshold}")
    logger.info(f"计算HD95: {'是' if enable_hd95 else '否'}")
    
    start_time = time.time()
    valid_samples = 0
    
    # 测试循环
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            try:
                # 统一的数据准备
                sam_batched_input, gt_mask, text_prompts = prepare_batch_data(batch_data, model_dtype)
                
                # 检查是否有GT标签
                if gt_mask is None:
                    logger.warning(f"Batch {batch_idx} 没有GT标签，跳过")
                    continue
                
                # 模型推理
                sam_output = model(
                    batched_input=sam_batched_input, 
                    multimask_output=True, 
                    text_prompts=text_prompts
                )
                
                # 获取原始尺寸和目标尺寸
                original_sizes = [sam_batched_input[i]['original_size'] for i in range(len(sam_batched_input))]
                target_size = gt_mask.shape[-2:]
                
                # 处理SAM输出
                pred_logits = process_sam_output_unified(
                    sam_output, 
                    original_sizes, 
                    target_size=target_size,
                    multimask_output=True, 
                    return_logits=True
                )
                
                # 转换为概率
                pred_probs = torch.sigmoid(pred_logits)
                
                # 计算批次指标
                batch_metrics = calculate_batch_metrics(
                    pred_probs, 
                    gt_mask, 
                    threshold=test_threshold,
                    input_is_logits=False,
                    enable_hd95=enable_hd95
                )
                
                # 收集指标
                for key in all_metrics.keys():
                    if batch_metrics[key] is not None:
                        all_metrics[key].append(batch_metrics[key])
                
                valid_samples += pred_probs.shape[0]
                
                # 进度显示
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"已处理 {batch_idx + 1}/{len(test_loader)} 批次")
                
                # 显存清理
                del sam_output, pred_logits, pred_probs
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"测试批次 {batch_idx} 出错: {str(e)}")
                continue
    
    # 计算最终统计结果
    test_time = time.time() - start_time
    
    # 汇总指标
    final_metrics = {}
    for key, values in all_metrics.items():
        if values:  # 如果有有效值
            final_metrics[f'{key}_mean'] = np.mean(values)
            final_metrics[f'{key}_std'] = np.std(values)
            final_metrics[f'{key}_min'] = np.min(values)
            final_metrics[f'{key}_max'] = np.max(values)
        else:
            final_metrics[f'{key}_mean'] = None
            final_metrics[f'{key}_std'] = None
            final_metrics[f'{key}_min'] = None
            final_metrics[f'{key}_max'] = None
    
    # 测试结果
    test_results = {
        'metrics': final_metrics,
        'test_samples': valid_samples,
        'test_time': test_time,
        'test_threshold': test_threshold,
        'enable_hd95': enable_hd95
    }
    
    # 打印详细结果
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    logger.info(f"测试样本数: {valid_samples}")
    logger.info(f"测试时间: {test_time:.2f}秒")
    logger.info(f"使用阈值: {test_threshold}")
    logger.info("-" * 60)
    
    # 打印各项指标
    metrics_to_show = ['iou', 'dice', 'precision', 'recall', 'f1', 'pa']
    if enable_hd95:
        metrics_to_show.append('hd95')
    
    for metric in metrics_to_show:
        mean_val = final_metrics[f'{metric}_mean']
        std_val = final_metrics[f'{metric}_std']
        min_val = final_metrics[f'{metric}_min']
        max_val = final_metrics[f'{metric}_max']
        
        if mean_val is not None:
            if metric == 'hd95':
                logger.info(f"{metric.upper():>10}: {mean_val:7.2f} ± {std_val:6.2f} (范围: {min_val:6.2f} - {max_val:6.2f})")
            else:
                logger.info(f"{metric.upper():>10}: {mean_val:7.4f} ± {std_val:6.4f} (范围: {min_val:6.4f} - {max_val:6.4f})")
        else:
            logger.info(f"{metric.upper():>10}: N/A")
    
    logger.info("=" * 60)
    
    return test_results


def test_integrated_with_thresholds(test_loader, model, cfg, 
                                  threshold_list: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    使用多个阈值进行测试，找到最佳阈值
    
    Args:
        test_loader: 测试数据加载器
        model: 训练好的模型
        cfg: 配置对象
        threshold_list: 阈值列表
        
    Returns:
        multi_threshold_results: 多阈值测试结果
    """
    logger.info(f"开始多阈值测试，阈值列表: {threshold_list}")
    
    multi_threshold_results = {}
    best_threshold = 0.5
    best_dice = 0.0
    
    for threshold in threshold_list:
        logger.info(f"\n--- 测试阈值: {threshold} ---")
        
        # 临时修改配置中的阈值
        original_threshold = getattr(cfg, 'test_threshold', 0.5)
        cfg.test_threshold = threshold
        
        # 进行测试
        result = test_integrated(test_loader, model, cfg, enable_hd95=False)  # 多阈值测试时不计算HD95节省时间
        
        # 恢复原始阈值
        cfg.test_threshold = original_threshold
        
        # 保存结果
        multi_threshold_results[f'threshold_{threshold}'] = result
        
        # 记录最佳阈值
        dice_mean = result['metrics']['dice_mean']
        if dice_mean is not None and dice_mean > best_dice:
            best_dice = dice_mean
            best_threshold = threshold
    
    # 使用最佳阈值进行完整测试（包含HD95）
    logger.info(f"\n使用最佳阈值 {best_threshold} 进行完整测试...")
    cfg.test_threshold = best_threshold
    best_result = test_integrated(test_loader, model, cfg, enable_hd95=True)
    
    # 汇总结果
    multi_threshold_results['best_threshold'] = best_threshold
    multi_threshold_results['best_result'] = best_result
    multi_threshold_results['threshold_comparison'] = {
        f'threshold_{t}': multi_threshold_results[f'threshold_{t}']['metrics']['dice_mean'] 
        for t in threshold_list
    }
    
    # 打印最佳阈值结果
    logger.info("=" * 60)
    logger.info("最佳阈值测试结果")
    logger.info("=" * 60)
    logger.info(f"最佳阈值: {best_threshold}")
    logger.info(f"各阈值Dice对比:")
    for t in threshold_list:
        dice = multi_threshold_results['threshold_comparison'][f'threshold_{t}']
        mark = " ← 最佳" if t == best_threshold else ""
        logger.info(f"  阈值 {t}: {dice:.4f}{mark}")
    
    return multi_threshold_results

def quick_test_integrated(test_loader, model, cfg, max_batches: int = 50):
    """
    快速测试函数 - 只测试部分数据，用于快速验证
    
    Args:
        test_loader: 测试数据加载器
        model: 模型
        cfg: 配置
        max_batches: 最大测试批次数
        
    Returns:
        quick_results: 快速测试结果
    """
    logger.info(f"开始快速测试，最多处理 {max_batches} 个批次")
    
    model.eval()
    model_dtype = next(model.parameters()).dtype
    test_threshold = getattr(cfg, 'test_threshold', 0.5)
    
    all_metrics = {'iou': [], 'dice': [], 'pa': []}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                sam_batched_input, gt_mask, text_prompts = prepare_batch_data(batch_data, model_dtype)
                
                if gt_mask is None:
                    continue
                
                sam_output = model(batched_input=sam_batched_input, multimask_output=True, text_prompts=text_prompts)
                original_sizes = [sam_batched_input[i]['original_size'] for i in range(len(sam_batched_input))]
                target_size = gt_mask.shape[-2:]
                
                pred_logits = process_sam_output_unified(
                    sam_output, original_sizes, target_size=target_size,
                    multimask_output=True, return_logits=True
                )
                
                pred_probs = torch.sigmoid(pred_logits)
                
                # 使用快速版本计算指标
                from utils.integrated_loss import calculate_batch_metrics_fast
                batch_metrics = calculate_batch_metrics_fast(pred_probs, gt_mask, input_is_logits=False)
                
                for key in all_metrics.keys():
                    all_metrics[key].append(batch_metrics[key])
                
                del sam_output, pred_logits, pred_probs
                
            except Exception as e:
                logger.error(f"快速测试批次 {batch_idx} 出错: {str(e)}")
                continue
    
    # 计算平均指标
    quick_results = {}
    for key, values in all_metrics.items():
        if values:
            quick_results[key] = np.mean(values)
        else:
            quick_results[key] = 0.0
    
    logger.info(f"快速测试结果 (阈值={test_threshold}):")
    logger.info(f"  IoU: {quick_results['iou']:.4f}")
    logger.info(f"  Dice: {quick_results['dice']:.4f}") 
    logger.info(f"  PA: {quick_results['pa']:.4f}")
    
    return quick_results

# 向后兼容的函数别名
def process_sam_output(sam_output, original_sizes, multimask_output=True, target_size=None, use_logits=True):
    """向后兼容的接口"""
    return process_sam_output_unified(
        sam_output, 
        original_sizes, 
        target_size=target_size or original_sizes[0],
        multimask_output=multimask_output,
        return_logits=use_logits
    )


def process_sam_output_with_logits(sam_output, original_sizes, multimask_output=True, target_size=None):
    """向后兼容的接口"""
    return process_sam_output_unified(
        sam_output, 
        original_sizes, 
        target_size=target_size or original_sizes[0],
        multimask_output=multimask_output,
        return_logits=True
    )