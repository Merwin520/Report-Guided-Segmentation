import torch
from typing import List, Dict, Any, Optional, Union
from segment_anything.modeling.sam import Sam
from .bridge import create_bridge
from .point_sampler import GumbelPointSampler
from .box_sampler import create_box_sampler


class FlexibleParameterManager:
    """灵活的参数管理器 - 精确识别可训练组件"""
    
    @classmethod
    def classify_parameter(cls, param_name: str) -> str:
        """精确的参数分类方法"""
        # 清理参数名，移除常见前缀
        clean_name = param_name
        for prefix in ['model.', 'module.']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        # 按优先级进行精确匹配
        
        # 1. DETRIS CoCoOp 组件（最具体）
        if any(x in clean_name for x in ['meta_net', 'ctx_embeddings', 'condition_projector']):
            return 'detris_cocoop'
        
        # 2. DETRIS 适配器（在backbone匹配之前）
        if 'detris.txt_backbone' in clean_name and 'adapter' in clean_name:
            return 'detris_text_adapter'
        if 'detris.dinov2' in clean_name and 'adapter' in clean_name:
            return 'detris_vision_adapter'
        
        # 3. SAM 适配器（在encoder匹配之前）
        if 'image_encoder' in clean_name and 'adapter' in clean_name:
            return 'sam_image_encoder_adapter'
        
        # 4. DETRIS 主要组件
        if 'detris.txt_backbone' in clean_name:
            return 'detris_text_encoder'
        if 'detris.dinov2' in clean_name:
            return 'detris_vision_encoder'
        if 'detris.fusion' in clean_name:
            return 'detris_fusion'
        if 'detris.neck' in clean_name:
            return 'detris_neck'
        if 'detris.decoder' in clean_name:
            return 'detris_decoder'
        if 'detris.proj' in clean_name:
            return 'detris_projector'
        
        # 5. SAM 主要组件
        if 'image_encoder.patch_embed' in clean_name:
            return 'sam_image_encoder_patch'
        if 'image_encoder.pos_embed' in clean_name:
            return 'sam_image_encoder_pos'
        if 'image_encoder.neck' in clean_name:
            return 'sam_image_encoder_neck'
        if 'image_encoder.blocks' in clean_name:
            return 'sam_image_encoder_blocks'
        if 'image_encoder' in clean_name:
            return 'sam_image_encoder'
        
        if 'prompt_encoder' in clean_name:
            return 'sam_prompt_encoder'
        if 'mask_decoder' in clean_name:
            return 'sam_mask_decoder'
        
        # 6. 其他组件
        if 'bridge' in clean_name:
            return 'bridge'
        if 'point_sampler' in clean_name:
            return 'point_sampler'
        
        return 'other'
    
    @classmethod
    def get_detailed_sam_adapter_info(cls, model) -> Dict[str, Any]:
        """详细获取SAM adapter信息"""
        adapter_info = {
            'has_adapters': False,
            'adapter_blocks': [],
            'adapter_parameters': {},
            'total_adapter_params': 0
        }
        
        if not hasattr(model, 'image_encoder') or not hasattr(model.image_encoder, 'blocks'):
            return adapter_info
        
        # 检查每个block
        for i, block in enumerate(model.image_encoder.blocks):
            if hasattr(block, 'adapter') and block.adapter is not None:
                adapter_info['has_adapters'] = True
                adapter_info['adapter_blocks'].append(i)
                
                # 统计该block adapter的参数
                block_params = list(block.adapter.parameters())
                if block_params:
                    adapter_name = f'sam_image_encoder_adapter_block_{i}'
                    adapter_info['adapter_parameters'][adapter_name] = block_params
                    adapter_info['total_adapter_params'] += sum(p.numel() for p in block_params)
        
        return adapter_info
    
    @classmethod
    def validate_classification(cls, model, debug=False) -> Dict[str, Any]:
        """验证参数分类的准确性"""
        classification_stats = {}
        unclassified_params = []
        
        for name, param in model.named_parameters():
            component = cls.classify_parameter(name)
            
            if component not in classification_stats:
                classification_stats[component] = {
                    'count': 0,
                    'total_params': 0,
                    'examples': []
                }
            
            classification_stats[component]['count'] += 1
            classification_stats[component]['total_params'] += param.numel()
            
            if len(classification_stats[component]['examples']) < 3:
                classification_stats[component]['examples'].append(name)
            
            if component == 'other':
                unclassified_params.append(name)
            
            if debug:
                print(f"{name:<50} -> {component}")
        
        return {
            'classification_stats': classification_stats,
            'unclassified_params': unclassified_params,
            'sam_adapter_info': cls.get_detailed_sam_adapter_info(model)
        }


class SamWithDetris(Sam):
    """简化的SamWithDetris - 完全由配置文件控制训练组件"""
    
    def __init__(
        self,
        detris_model,
        image_encoder, 
        prompt_encoder,
        mask_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        
        # 采样配置
        use_point_sampling: bool = True,
        use_box_sampling: bool = True,
        point_sampler_config: Optional[Dict[str, Any]] = None,
        box_sampler_config: Optional[Dict[str, Any]] = None,
        bridge_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder, 
            mask_decoder=mask_decoder,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std
        )
        
        self.detris = detris_model
        self.param_manager = FlexibleParameterManager()
        
        # 初始化采样组件
        self.use_point_sampling = use_point_sampling
        self.use_box_sampling = use_box_sampling
        
        if use_point_sampling:
            default_point_config = {
                'temperature': 1.0,
                'hard': False,
                'num_points': 1,
                'min_confidence': 0.1
            }
            final_point_config = {**default_point_config, **(point_sampler_config or {})}
            self.point_sampler = GumbelPointSampler(**final_point_config)
        else:
            self.point_sampler = None
        
        # 初始化bridge
        default_bridge_config = {
            'bridge_type': 'trainable',  # 使用简单bridge，无可训练参数
            'target_size': (256, 256),
            'use_box_sampling': use_box_sampling,
            'box_sampler_config': box_sampler_config or {}
        }
        final_bridge_config = {**default_bridge_config, **(bridge_config or {})}
        self.bridge = create_bridge(**final_bridge_config)
        
        # 初始状态：冻结所有参数
        self.freeze_all_parameters()
        
        print("SamWithDetris initialized - all parameters frozen by default")
        print("Use set_trainable_components() with your config to enable training")
        self._print_component_summary()
    
    def freeze_all_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def set_trainable_components(self, trainable_config: Dict[str, bool]):
        """
        根据配置设置可训练组件
        
        Args:
            trainable_config: 组件训练配置，例如：
            {
                'detris_text_adapter': True,
                'detris_vision_adapter': True,
                'detris_projector': True,
                'sam_image_encoder_adapter': True,  # SAM adapter
                'sam_mask_decoder': True,
                'sam_prompt_encoder': False,
                ...
            }
        """
        print(f"Setting trainable components based on config...")
        
        # 首先冻结所有参数
        self.freeze_all_parameters()
        
        # 统计信息
        component_stats = {}
        total_trainable = 0
        
        # 处理常规参数
        for name, param in self.named_parameters():
            component = self.param_manager.classify_parameter(name)
            
            if trainable_config.get(component, False):
                param.requires_grad = True
                total_trainable += param.numel()
                
                if component not in component_stats:
                    component_stats[component] = {'params': 0, 'count': 0}
                component_stats[component]['params'] += param.numel()
                component_stats[component]['count'] += 1
        
        # 特殊处理SAM adapter
        if trainable_config.get('sam_image_encoder_adapter', False):
            adapter_info = self.param_manager.get_detailed_sam_adapter_info(self)
            if adapter_info['has_adapters']:
                for adapter_name, adapter_params in adapter_info['adapter_parameters'].items():
                    for param in adapter_params:
                        if not param.requires_grad:  # 避免重复计数
                            param.requires_grad = True
                            total_trainable += param.numel()
                            
                            if adapter_name not in component_stats:
                                component_stats[adapter_name] = {'params': 0, 'count': 0}
                            component_stats[adapter_name]['params'] += param.numel()
                            component_stats[adapter_name]['count'] += 1
            else:
                print("Warning: sam_image_encoder_adapter requested but no adapters found")
        
        # 打印结果
        print(f"\nTrainable components configured:")
        for component, stats in component_stats.items():
            print(f"  - {component}: {stats['count']} groups, {stats['params']:,} params")
        
        print(f"Total trainable parameters: {total_trainable:,}")
        
        return component_stats
    
    def get_trainable_parameters(self, separate_by_component=False):
        """获取可训练参数"""
        if not separate_by_component:
            return [p for p in self.parameters() if p.requires_grad]
        
        param_groups = {}
        
        # 常规参数分组
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            component = self.param_manager.classify_parameter(name)
            if component not in param_groups:
                param_groups[component] = []
            param_groups[component].append(param)
        
        # SAM adapter单独处理
        adapter_info = self.param_manager.get_detailed_sam_adapter_info(self)
        if adapter_info['has_adapters']:
            for adapter_name, adapter_params in adapter_info['adapter_parameters'].items():
                trainable_adapter_params = [p for p in adapter_params if p.requires_grad]
                if trainable_adapter_params:
                    param_groups[adapter_name] = trainable_adapter_params
        
        return param_groups
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        trainable_params = self.get_trainable_parameters(separate_by_component=True)
        
        return {
            'total_trainable_params': sum(sum(p.numel() for p in params) 
                                        for params in trainable_params.values()),
            'trainable_components': list(trainable_params.keys()),
            'sampling_capabilities': {
                'point_sampling': self.point_sampler is not None,
                'box_sampling': hasattr(self.bridge, 'box_sampler') and 
                            getattr(self.bridge, 'box_sampler', None) is not None,
                'mask_processing': True
            },
            'bridge_type': getattr(self.bridge, 'bridge_type', 'default') if hasattr(self, 'bridge') else 'none'
        }

    def setup_optimizer_param_groups(self, lr_config: Dict[str, float]):
        """
        设置优化器参数组 - 避免参数重复
        """
        trainable_params = self.get_trainable_parameters(separate_by_component=True)
        param_groups = []
        used_params = set()  # 跟踪已使用的参数
        
        print("Optimizer parameter groups:")
        total_params = 0
        
        # 优先处理具体的adapter blocks，避免与通用adapter重叠
        adapter_blocks = [k for k in trainable_params.keys() if k.startswith('sam_image_encoder_adapter_block_')]
        
        for component in adapter_blocks:
            params = trainable_params[component]
            if not params:
                continue
                
            # 检查学习率配置
            lr = lr_config.get(component)
            if lr is None:
                # 尝试使用通用adapter学习率
                lr = lr_config.get('sam_image_encoder_adapter')
            
            if lr is not None:
                # 检查参数是否已被使用
                param_ids = set(id(p) for p in params)
                if not param_ids.intersection(used_params):
                    param_count = sum(p.numel() for p in params)
                    param_groups.append({
                        'params': params,
                        'lr': lr,
                        'name': component
                    })
                    used_params.update(param_ids)
                    print(f"  - {component}: lr={lr}, {param_count:,} params")
                    total_params += param_count
        
        # 处理其他组件，跳过通用的sam_image_encoder_adapter以避免重复
        for component, params in trainable_params.items():
            if (component in adapter_blocks or 
                component == 'sam_image_encoder_adapter' or  # 跳过通用adapter
                not params):
                continue
                
            # 检查学习率配置
            lr = lr_config.get(component)
            if lr is not None:
                # 检查参数是否已被使用
                param_ids = set(id(p) for p in params)
                if not param_ids.intersection(used_params):
                    param_count = sum(p.numel() for p in params)
                    param_groups.append({
                        'params': params,
                        'lr': lr,
                        'name': component
                    })
                    used_params.update(param_ids)
                    print(f"  - {component}: lr={lr}, {param_count:,} params")
                    total_params += param_count
            else:
                print(f"  Warning: {component} has trainable params but no lr config")
        
        if not param_groups:
            raise ValueError("No parameter groups configured! Check lr_config.")
        
        print(f"Total: {len(param_groups)} groups, {total_params:,} parameters")
        return param_groups
    
    def get_available_components(self) -> Dict[str, int]:
        """获取所有可用的组件及其参数数量"""
        components = {}
        
        # 统计常规组件
        for name, param in self.named_parameters():
            component = self.param_manager.classify_parameter(name)
            if component not in components:
                components[component] = 0
            components[component] += param.numel()
        
        # 统计SAM adapter（避免重复计数）
        adapter_info = self.param_manager.get_detailed_sam_adapter_info(self)
        if adapter_info['has_adapters']:
            # 从sam_image_encoder中减去adapter的参数，避免重复计数
            total_adapter_params = adapter_info['total_adapter_params']
            if 'sam_image_encoder' in components:
                components['sam_image_encoder'] -= total_adapter_params
            
            # 添加具体的adapter组件
            for adapter_name in adapter_info['adapter_parameters']:
                adapter_params = adapter_info['adapter_parameters'][adapter_name]
                components[adapter_name] = sum(p.numel() for p in adapter_params)
        
        return components
    
    def validate_parameter_classification(self, debug=False):
        """验证参数分类准确性"""
        print("Validating parameter classification...")
        validation_results = self.param_manager.validate_classification(self, debug=debug)
        
        print(f"\nClassification Summary:")
        for component, stats in validation_results['classification_stats'].items():
            print(f"  - {component}: {stats['count']} parameters, {stats['total_params']:,} values")
        
        if validation_results['unclassified_params']:
            print(f"\nUnclassified parameters ({len(validation_results['unclassified_params'])}):")
            for param in validation_results['unclassified_params'][:5]:  # 只显示前5个
                print(f"  - {param}")
            if len(validation_results['unclassified_params']) > 5:
                print(f"  ... and {len(validation_results['unclassified_params']) - 5} more")
        
        adapter_info = validation_results['sam_adapter_info']
        if adapter_info['has_adapters']:
            print(f"\nSAM Adapter Info:")
            print(f"  - Adapter blocks: {adapter_info['adapter_blocks']}")
            print(f"  - Total adapter parameters: {adapter_info['total_adapter_params']:,}")
            for adapter_name in adapter_info['adapter_parameters']:
                params = adapter_info['adapter_parameters'][adapter_name]
                print(f"  - {adapter_name}: {len(params)} parameter tensors")
        else:
            print("\nNo SAM adapters found")
        
        return validation_results
    
    def _print_component_summary(self):
        """打印组件摘要"""
        components = self.get_available_components()
        
        print(f"\nAvailable components:")
        detris_components = {k: v for k, v in components.items() if k.startswith('detris_')}
        sam_components = {k: v for k, v in components.items() if k.startswith('sam_')}
        other_components = {k: v for k, v in components.items() if not k.startswith(('detris_', 'sam_'))}
        
        if detris_components:
            print("  DETRIS components:")
            for comp, params in detris_components.items():
                print(f"    - {comp}: {params:,} params")
        
        if sam_components:
            print("  SAM components:")
            for comp, params in sam_components.items():
                print(f"    - {comp}: {params:,} params")
        
        if other_components:
            print("  Other components:")
            for comp, params in other_components.items():
                print(f"    - {comp}: {params:,} params")
    
    def print_parameter_stats(self):
        """打印参数统计"""
        components = self.get_available_components()
        trainable_params = self.get_trainable_parameters(separate_by_component=True)
        
        print(f"\nParameter Statistics:")
        print(f"{'Component':<35} {'Total':<12} {'Trainable':<12} {'Status'}")
        print("-" * 70)
        
        total_all = 0
        total_trainable = 0
        
        for component, total_params in sorted(components.items()):
            trainable_count = sum(p.numel() for p in trainable_params.get(component, []))
            status = "✓" if trainable_count > 0 else "✗"
            
            print(f"{component:<35} {total_params:<12,} {trainable_count:<12,} {status}")
            total_all += total_params
            total_trainable += trainable_count
        
        print("-" * 70)
        print(f"{'TOTAL':<35} {total_all:<12,} {total_trainable:<12,} ({total_trainable/total_all*100:.1f}%)")
    
    # ========== 采样控制 ==========
    
    def set_point_sampling(self, enabled: bool, **config):
        """控制点采样"""
        self.use_point_sampling = enabled
        if enabled and self.point_sampler is not None:
            for key, value in config.items():
                if hasattr(self.point_sampler, key):
                    setattr(self.point_sampler, key, value)
        print(f"Point sampling: {'enabled' if enabled else 'disabled'}")
    
    def set_box_sampling(self, enabled: bool, **config):
        """控制box采样"""
        self.use_box_sampling = enabled
        if hasattr(self.bridge, 'set_box_sampling'):
            self.bridge.set_box_sampling(enabled)
        print(f"Box sampling: {'enabled' if enabled else 'disabled'}")
    
    def get_sampling_capabilities(self) -> Dict[str, bool]:
        """获取采样能力"""
        return {
            'point_sampling': self.point_sampler is not None,
            'box_sampling': (hasattr(self.bridge, 'box_sampler') and 
                        self.bridge.box_sampler is not None),
            'mask_processing': True,
            'multi_prompt': (self.point_sampler is not None or 
                        (hasattr(self.bridge, 'box_sampler') and 
                            self.bridge.box_sampler is not None))
        }

    # ========== 核心前向传播 ==========
    
    def _extract_images_for_detris(self, batched_input: List[Dict[str, Any]]) -> torch.Tensor:
        """提取DETRIS图像"""
        images = []
        for input_dict in batched_input:
            img = input_dict['image']
            if img.shape[-2:] != (518, 518):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(518, 518), mode='bilinear', align_corners=False
                ).squeeze(0)
            images.append(img)
        return torch.stack(images, dim=0)
    
    def _tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """文本tokenization"""
        if hasattr(self.detris, 'txt_backbone') and hasattr(self.detris.txt_backbone, 'tokenizer'):
            tokenizer = self.detris.txt_backbone.tokenizer
            max_length = 77
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized')
            max_length = 77
            
        encoded = tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt'
        )
        device = next(self.parameters()).device
        return encoded['input_ids'].to(device)
    
    def forward(
        self,
        batched_input: List[Dict[str, Any]], 
        multimask_output: bool,
        text_prompts: Optional[List[str]] = None,
        enable_point_sampling: Optional[bool] = None,
        enable_box_sampling: Optional[bool] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """前向传播"""
        
        # 采样配置
        use_points = (enable_point_sampling if enable_point_sampling is not None 
                     else self.use_point_sampling)
        use_boxes = (enable_box_sampling if enable_box_sampling is not None 
                    else self.use_box_sampling)
        
        if text_prompts is None:
            text_prompts = [""] * len(batched_input)
        
        # DETRIS前向传播
        detris_images = self._extract_images_for_detris(batched_input)
        word_tokens = self._tokenize_texts(text_prompts)
        
        # 根据DETRIS是否有可训练参数决定模式
        detris_has_trainable = any(p.requires_grad for p in self.detris.parameters())
        
        if self.training and detris_has_trainable:
            self.detris.train()
            detris_masks = self.detris(detris_images, word_tokens, mask=None)
        else:
            self.detris.eval()
            with torch.no_grad():
                detris_masks = self.detris(detris_images, word_tokens)
        
        # Bridge处理
        prompt_results = self.bridge.forward_with_prompts(
            detris_masks,
            generate_boxes_from_mask=use_boxes
        )
        
        sam_mask_inputs = prompt_results['mask_inputs'] 
        sam_box_coords = prompt_results.get('boxes', None) if use_boxes else None
        
        # 准备SAM输入
        for i in range(len(batched_input)):
            batched_input[i]['mask_inputs'] = sam_mask_inputs[i].unsqueeze(0)
            
            if use_points and self.point_sampler is not None:
                point_coords, point_labels = self.point_sampler(sam_mask_inputs[i:i+1])
                batched_input[i]['point_coords'] = point_coords
                batched_input[i]['point_labels'] = point_labels
            
            if use_boxes and sam_box_coords is not None:
                batched_input[i]['boxes'] = sam_box_coords[i:i+1]
        
        return super().forward(batched_input, multimask_output)
    
    def inference_with_text(
        self, 
        images: torch.Tensor,
        text_prompts: List[str],
        original_sizes: List[tuple],
        multimask_output: bool = True,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """文本推理接口"""
        batched_input = []
        for i in range(len(images)):
            batched_input.append({
                'image': images[i],
                'original_size': original_sizes[i]
            })
        return self.forward(batched_input, multimask_output, text_prompts, **kwargs)


class SamWithDetrisPredictor:
    """预测器"""
    
    def __init__(self, sam_detris_model: SamWithDetris):
        self.model = sam_detris_model
        self.device = next(self.model.parameters()).device
        self.reset_image()
    
    def set_image(self, image: torch.Tensor, text_prompt: str = ""):
        if image.dim() == 3 and image.shape[0] == 3:
            self.image = image.to(self.device)
        elif image.dim() == 3 and image.shape[2] == 3:
            self.image = image.permute(2, 0, 1).to(self.device)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
            
        self.original_size = self.image.shape[-2:]
        self.text_prompt = text_prompt
        self.is_image_set = True
    
    def predict(self, multimask_output: bool = True, **kwargs) -> tuple:
        if not self.is_image_set:
            raise RuntimeError("Image must be set before prediction")
        
        results = self.model.inference_with_text(
            images=self.image.unsqueeze(0),
            text_prompts=[self.text_prompt],
            original_sizes=[self.original_size],
            multimask_output=multimask_output,
            **kwargs
        )
        
        result = results[0]
        masks = result['masks'].squeeze(0)
        scores = result['iou_predictions'].squeeze(0) 
        logits = result['low_res_logits'].squeeze(0)
        return masks, scores, logits
    
    def reset_image(self):
        self.image = None
        self.text_prompt = ""
        self.original_size = None
        self.is_image_set = False


def create_sam_with_detris(
    detris_model,
    sam_components: Dict[str, Any],
    **kwargs
) -> SamWithDetris:
    """创建SamWithDetris模型"""
    return SamWithDetris(
        detris_model=detris_model,
        image_encoder=sam_components['image_encoder'],
        prompt_encoder=sam_components['prompt_encoder'], 
        mask_decoder=sam_components['mask_decoder'],
        **kwargs
    )


if __name__ == "__main__":
    print("Flexible SamWithDetris - Configuration-driven Training")
    print("="*60)
    print("Features:")
    print("  1. No predefined training strategies")
    print("  2. Fully configuration-driven component training")
    print("  3. SAM adapter support") 
    print("  4. Simple bridge (no trainable parameters)")
    print("  5. Flexible parameter grouping for optimizers")
    print("\nUsage:")
    print("  1. Create model: model = SamWithDetris(...)")
    print("  2. Check components: model.get_available_components()")
    print("  3. Set trainable: model.set_trainable_components(config)")
    print("  4. Setup optimizer: model.setup_optimizer_param_groups(lr_config)")