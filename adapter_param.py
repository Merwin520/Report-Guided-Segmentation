#!/usr/bin/env python3
"""
Adapter Parameter Analyzer
专门用于分析模型中adapter参数的工具脚本，无需训练即可查看参数统计
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from loguru import logger
from tabulate import tabulate
import json

# 添加项目路径（根据实际情况调整）
# sys.path.append('/path/to/your/project')

import utils.config as config
from model import build_segmenter


def setup_logger():
    """设置日志输出"""
    logger.remove()  # 移除默认handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )


class AdapterAnalyzer:
    """Adapter参数分析器"""
    
    def __init__(self):
        self.adapter_keywords = [
            'adapter', 'lora', 'prompt', 'prefix', 'cocoop', 
            'context', 'text_encoder', 'learnable', 'bottleneck',
            'linear_adapter', 'mlp_adapter', 'attention_adapter',
            'scale', 'shift', 'bias_adapter', 'norm_adapter'
        ]
        
        self.backbone_keywords = [
            'backbone', 'encoder', 'resnet', 'vit', 'swin', 
            'convnext', 'clip', 'visual', 'transformer',
            'bert', 'roberta', 'gpt', 'llama'
        ]
        
        self.head_keywords = [
            'head', 'classifier', 'decoder', 'predictor',
            'segmentation_head', 'detection_head', 'mlp_head'
        ]
    
    def is_adapter_param(self, name):
        """判断参数是否属于adapter"""
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.adapter_keywords)
    
    def is_backbone_param(self, name):
        """判断参数是否属于backbone"""
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.backbone_keywords)
    
    def is_head_param(self, name):
        """判断参数是否属于head"""
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in self.head_keywords)
    
    def get_param_type(self, name):
        """获取参数类型"""
        if self.is_adapter_param(name):
            return 'adapter'
        elif self.is_backbone_param(name):
            return 'backbone'
        elif self.is_head_param(name):
            return 'head'
        else:
            return 'other'
    
    def format_params(self, num_params):
        """格式化参数数量显示"""
        if num_params >= 1e9:
            return f"{num_params/1e9:.2f}B"
        elif num_params >= 1e6:
            return f"{num_params/1e6:.2f}M"
        elif num_params >= 1e3:
            return f"{num_params/1e3:.2f}K"
        else:
            return f"{num_params}"
    
    def analyze_model(self, model, verbose=False):
        """分析模型参数"""
        # 获取实际模型（处理DDP包装）
        actual_model = model.module if hasattr(model, 'module') else model
        
        # 初始化统计数据
        stats = {
            'total_params': 0,
            'trainable_params': 0,
            'frozen_params': 0,
            'adapter_params': 0,
            'adapter_trainable': 0,
            'adapter_frozen': 0,
            'backbone_params': 0,
            'backbone_trainable': 0,
            'backbone_frozen': 0,
            'head_params': 0,
            'head_trainable': 0,
            'head_frozen': 0,
            'other_params': 0,
            'other_trainable': 0,
            'other_frozen': 0,
            'param_details': []
        }
        
        # 遍历所有参数
        for name, param in actual_model.named_parameters():
            param_count = param.numel()
            is_trainable = param.requires_grad
            param_type = self.get_param_type(name)
            
            # 总参数统计
            stats['total_params'] += param_count
            if is_trainable:
                stats['trainable_params'] += param_count
            else:
                stats['frozen_params'] += param_count
            
            # 分类统计
            stats[f'{param_type}_params'] += param_count
            if is_trainable:
                stats[f'{param_type}_trainable'] += param_count
            else:
                stats[f'{param_type}_frozen'] += param_count
            
            # 记录详细信息
            stats['param_details'].append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count,
                'trainable': is_trainable,
                'type': param_type,
                'dtype': str(param.dtype)
            })
        
        return stats
    
    def print_summary(self, stats):
        """打印参数统计摘要"""
        logger.info("=" * 80)
        logger.info("🔍 ADAPTER PARAMETER ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        # 总体统计
        logger.info("📊 OVERALL STATISTICS:")
        logger.info(f"  Total Parameters:     {self.format_params(stats['total_params']):>10} ({stats['total_params']:,})")
        logger.info(f"  Trainable Parameters: {self.format_params(stats['trainable_params']):>10} ({stats['trainable_params']:,})")
        logger.info(f"  Frozen Parameters:    {self.format_params(stats['frozen_params']):>10} ({stats['frozen_params']:,})")
        
        if stats['total_params'] > 0:
            trainable_ratio = stats['trainable_params'] / stats['total_params'] * 100
            logger.info(f"  Trainable Ratio:      {trainable_ratio:>9.2f}%")
        
        logger.info("")
        
        # 组件分解
        logger.info("🔧 COMPONENT BREAKDOWN:")
        
        components = ['adapter', 'backbone', 'head', 'other']
        for comp in components:
            total_key = f'{comp}_params'
            trainable_key = f'{comp}_trainable'
            frozen_key = f'{comp}_frozen'
            
            if stats[total_key] > 0:
                emoji = {'adapter': '🎯', 'backbone': '🏗️', 'head': '🎭', 'other': '⚙️'}[comp]
                comp_ratio = stats[total_key] / stats['total_params'] * 100
                train_ratio = stats[trainable_key] / stats[total_key] * 100 if stats[total_key] > 0 else 0
                frozen_ratio = stats[frozen_key] / stats[total_key] * 100 if stats[total_key] > 0 else 0
                
                logger.info(f"  {emoji} {comp.capitalize()} Parameters:")
                logger.info(f"    Total:     {self.format_params(stats[total_key]):>10} ({comp_ratio:.2f}% of total)")
                logger.info(f"    Trainable: {self.format_params(stats[trainable_key]):>10} ({train_ratio:.2f}% of {comp})")
                logger.info(f"    Frozen:    {self.format_params(stats[frozen_key]):>10} ({frozen_ratio:.2f}% of {comp})")
                logger.info("")
        
        # 效率分析
        if stats['adapter_params'] > 0:
            logger.info("⚡ EFFICIENCY ANALYSIS:")
            if stats['backbone_params'] > 0:
                efficiency = stats['adapter_trainable'] / stats['backbone_params'] * 100
                logger.info(f"  Adapter-to-Backbone Ratio: {efficiency:.4f}%")
            
            adapter_efficiency = stats['adapter_trainable'] / stats['total_params'] * 100
            logger.info(f"  Adapter-to-Total Ratio:    {adapter_efficiency:.4f}%")
            
            if stats['adapter_trainable'] > 0:
                memory_eff = "High" if adapter_efficiency < 1 else "Medium" if adapter_efficiency < 5 else "Low"
                logger.info(f"  Memory Efficiency:         {memory_eff}")
        
        logger.info("=" * 80)
    
    def print_detailed_table(self, stats, component_filter=None, max_rows=50):
        """打印详细的参数表格"""
        details = stats['param_details']
        
        # 过滤组件
        if component_filter:
            details = [d for d in details if d['type'] == component_filter]
        
        if not details:
            logger.warning(f"No parameters found for component: {component_filter}")
            return
        
        # 按参数数量排序
        details = sorted(details, key=lambda x: x['params'], reverse=True)
        
        # 限制显示行数
        if len(details) > max_rows:
            logger.info(f"Showing top {max_rows} parameters (total: {len(details)})")
            details = details[:max_rows]
        
        # 准备表格数据
        headers = ["Parameter Name", "Shape", "Params", "Trainable", "Type", "Dtype"]
        table_data = []
        
        for detail in details:
            table_data.append([
                detail['name'][:60] + ('...' if len(detail['name']) > 60 else ''),  # 截断长名称
                str(detail['shape']),
                self.format_params(detail['params']),
                "✓" if detail['trainable'] else "✗",
                detail['type'],
                detail['dtype']
            ])
        
        logger.info(f"\n📋 DETAILED PARAMETER TABLE ({component_filter or 'all'}):")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def save_json_report(self, stats, output_path):
        """保存JSON格式的详细报告"""
        report = {
            'summary': {k: v for k, v in stats.items() if k != 'param_details'},
            'detailed_parameters': stats['param_details']
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Detailed report saved to: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Adapter Parameter Analyzer')
    parser.add_argument('--config', required=True, help='Model config file path')
    parser.add_argument('--weights', type=str, default=None, help='Pretrained weights path (optional)')
    parser.add_argument('--component', type=str, choices=['adapter', 'backbone', 'head', 'other', 'all'], 
                        default='all', help='Component to analyze in detail')
    parser.add_argument('--max-rows', type=int, default=50, help='Maximum rows to display in table')
    parser.add_argument('--save-json', type=str, default=None, help='Save detailed report to JSON file')
    parser.add_argument('--no-table', action='store_true', help='Skip detailed table output')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None,
                        help='Override config options')
    
    return parser.parse_args()


def load_config(config_path, opts=None):
    """加载配置文件"""
    cfg = config.load_cfg_from_cfg_file(config_path)
    if opts:
        cfg = config.merge_cfg_from_list(cfg, opts)
    
    # 展平嵌套配置
    flat_config = {}
    for section in ['DATA', 'TRAIN', 'COCOOP', 'CONTRASTIVE', 'LOSS', 'TEST', 'MISC']:
        if hasattr(cfg, section):
            section_cfg = getattr(cfg, section)
            if hasattr(section_cfg, 'items'):
                for key, value in section_cfg.items():
                    flat_config[key] = value
    
    for key, value in flat_config.items():
        setattr(cfg, key, value)
    
    return cfg


def load_pretrained_weights(model, weights_path):
    """加载预训练权重"""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    logger.info(f"Loading weights from: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 提取状态字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理DDP命名
    model_state_dict = {}
    model_has_module = hasattr(model, 'module')
    
    for key, value in state_dict.items():
        if key.startswith('module.') and not model_has_module:
            new_key = key[7:]  # 移除 'module.'
            model_state_dict[new_key] = value
        elif not key.startswith('module.') and model_has_module:
            new_key = f'module.{key}'
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    logger.info("✓ Weights loaded successfully")


def main():
    """主函数"""
    args = parse_args()
    setup_logger()
    
    try:
        logger.info("🔍 Starting Adapter Parameter Analysis")
        logger.info(f"📝 Config: {args.config}")
        if args.weights:
            logger.info(f"⚖️  Weights: {args.weights}")
        
        # 加载配置
        cfg = load_config(args.config, args.opts)
        
        # 构建模型
        logger.info("🏗️  Building model...")
        model_result = build_segmenter(cfg)
        
        # 处理可能的元组返回
        if isinstance(model_result, tuple):
            model = model_result[0]
            logger.info(f"Model builder returned tuple, using first element")
        else:
            model = model_result
        
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
        
        # 加载权重（如果指定）
        if args.weights:
            load_pretrained_weights(model, args.weights)
        
        # 设置为评估模式
        model.eval()
        
        # 创建分析器并分析
        analyzer = AdapterAnalyzer()
        logger.info("📊 Analyzing model parameters...")
        stats = analyzer.analyze_model(model, verbose=True)
        
        # 打印摘要
        analyzer.print_summary(stats)
        
        # 打印详细表格
        if not args.no_table:
            if args.component == 'all':
                for comp in ['adapter', 'backbone', 'head', 'other']:
                    comp_details = [d for d in stats['param_details'] if d['type'] == comp]
                    if comp_details:
                        analyzer.print_detailed_table(stats, comp, args.max_rows)
            else:
                analyzer.print_detailed_table(stats, args.component, args.max_rows)
        
        # 保存JSON报告
        if args.save_json:
            analyzer.save_json_report(stats, args.save_json)
        
        # 简要总结
        logger.info("\n🎯 QUICK SUMMARY:")
        logger.info(f"Adapter Parameters: {analyzer.format_params(stats['adapter_params'])} total, {analyzer.format_params(stats['adapter_trainable'])} trainable")
        logger.info(f"Training Efficiency: {stats['adapter_trainable']/stats['total_params']*100:.3f}% of total parameters")
        
        logger.info("✅ Analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()