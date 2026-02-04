import torch
from typing import List, Dict, Any, Optional

from segment_anything.modeling.sam import Sam
from .bridge import create_bridge
from .point_sampler import GumbelPointSampler


class SamWithDetris(Sam):
    """
    集成DETRIS pipeline的SAM模型 - 适配build_sam.py的SAM版本
    """
    
    def __init__(
        self,
        detris_model,
        image_encoder,
        prompt_encoder, 
        mask_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        bridge_config: Optional[Dict[str, Any]] = None,
        point_sampler_config: Optional[Dict[str, Any]] = None,
        use_point_sampling: bool = True
    ):
        super().__init__(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std
        )

        # 集成DETRIS并完全冻结
        self.detris = detris_model
        self._freeze_detris_completely()
        
        # 检查模型是否有adapter
        self.has_adapter = self._check_adapter_existence()
        
        # 初始化bridge
        if bridge_config is None:
            bridge_config = {'bridge_type': 'trainable'}
        self.bridge = create_bridge(**bridge_config)
        
        # 初始化点采样器
        self.use_point_sampling = use_point_sampling
        if use_point_sampling:
            if point_sampler_config is None:
                point_sampler_config = {
                    'temperature': 1.0,
                    'hard': False,
                    'num_points': 1,
                    'min_confidence': 0.1
                }
            self.point_sampler = GumbelPointSampler(**point_sampler_config)
        else:
            self.point_sampler = None
        
        print(f"SamWithDetris initialized:")
        print(f"  - DETRIS parameters: FROZEN")
        print(f"  - Bridge type: {bridge_config.get('bridge_type', 'default')}")
        print(f"  - Point sampling: {'enabled' if use_point_sampling else 'disabled'}")
    
    def _check_adapter_existence(self):
        """检查模型中是否存在adapter参数"""
        for name, param in self.image_encoder.named_parameters():
            if 'adapter' in name:
                return True
        return False
    
    def _freeze_detris_completely(self):
        """完全冻结DETRIS所有参数"""
        for param in self.detris.parameters():
            param.requires_grad = False
        self.detris.eval()
        print("DETRIS所有参数已完全冻结")
    
    def apply_sam_training_strategy(self):
        """应用SAM训练策略"""
        self._freeze_detris_completely()
        
        # 解冻SAM组件
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        
        print("SAM组件已解冻用于训练")
    
    def _extract_images_for_detris(self, batched_input: List[Dict[str, Any]]) -> torch.Tensor:
        """从批量输入中提取图像用于DETRIS"""
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
            encoded = tokenizer(
                texts, padding='max_length', truncation=True,
                max_length=77, return_tensors='pt'
            )
            return encoded['input_ids'].to(self.device)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized')
            encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            device = next(self.parameters()).device
            return encoded['input_ids'].to(device)
    
    def _generate_detris_masks(self, images: torch.Tensor, word_tokens: torch.Tensor) -> torch.Tensor:
        """生成DETRIS masks"""
        self.detris.eval()
        with torch.no_grad():
            detris_output = self.detris(images, word_tokens)
        return detris_output
    
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        text_prompts: Optional[List[str]] = None,
        enable_point_sampling: Optional[bool] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        适配build_sam.py的SAM版本
        输入：List[Dict[str, Any]]
        输出：List[Dict[str, torch.Tensor]]
        """
        use_points = enable_point_sampling if enable_point_sampling is not None else self.use_point_sampling
        use_points = use_points and self.point_sampler is not None
        
        if text_prompts is None:
            text_prompts = [""] * len(batched_input)
        
        # DETRIS前向传播
        detris_images = self._extract_images_for_detris(batched_input)
        word_tokens = self._tokenize_texts(text_prompts)
        detris_masks = self._generate_detris_masks(detris_images, word_tokens)
        
        # Bridge处理
        if use_points:
            sam_mask_inputs, sam_point_coords = self.bridge.forward_with_points(detris_masks, None)
        else:
            sam_mask_inputs = self.bridge(detris_masks)
            sam_point_coords = None
        
        # 点采样（如果启用）
        sam_point_labels = None
        if use_points and self.point_sampler is not None:
            point_coords, point_labels = self.point_sampler(sam_mask_inputs)
            sam_point_coords = point_coords
            sam_point_labels = point_labels
        
        # 为每个样本准备SAM输入
        sam_batched_input = []
        for i in range(len(batched_input)):
            sam_input = {
                'image': batched_input[i]['image'],
                'original_size': batched_input[i]['original_size'],
                'mask_inputs': sam_mask_inputs[i:i+1],  # 添加batch维度
            }
            
            # 添加点坐标（如果有）
            if sam_point_coords is not None:
                sam_input['point_coords'] = sam_point_coords[i:i+1]
                sam_input['point_labels'] = sam_point_labels[i:i+1] if sam_point_labels is not None else None
            
            sam_batched_input.append(sam_input)
        
        # 调用原始SAM的forward方法
        # 注意：需要临时移除@torch.no_grad()装饰器的影响
        if self.training:
            # 训练模式：手动调用SAM的forward实现，避开@torch.no_grad()
            return self._sam_forward_without_no_grad(sam_batched_input, multimask_output)
        else:
            # 推理模式：使用原始SAM的forward
            return super().forward(sam_batched_input, multimask_output)
    
    def _sam_forward_without_no_grad(
        self, 
        batched_input: List[Dict[str, Any]], 
        multimask_output: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """
        SAM的forward实现，但不使用@torch.no_grad()装饰器
        这是从原始SAM.forward方法复制的代码，去掉了@torch.no_grad()
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs
    
    def forward_single(
        self,
        image: torch.Tensor,
        original_size: tuple,
        text_prompt: str = "",
        multimask_output: bool = True,
        enable_point_sampling: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        单样本前向传播
        """
        # 构建单样本的batched_input
        batched_input = [{
            'image': image,
            'original_size': original_size
        }]
        
        # 调用forward方法
        results = self.forward(
            batched_input=batched_input,
            multimask_output=multimask_output,
            text_prompts=[text_prompt],
            enable_point_sampling=enable_point_sampling
        )
        
        # 返回第一个（也是唯一的）结果
        return results[0]
    
    def inference_with_text(
        self, 
        images: torch.Tensor, 
        text_prompts: List[str],
        original_sizes: List[tuple],
        multimask_output: bool = True,
        enable_point_sampling: Optional[bool] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """推理接口，保持兼容性"""
        results = []
        for i in range(len(images)):
            result = self.forward_single(
                image=images[i],
                original_size=original_sizes[i],
                text_prompt=text_prompts[i] if i < len(text_prompts) else "",
                multimask_output=multimask_output,
                enable_point_sampling=enable_point_sampling
            )
            results.append(result)
        return results


class SamWithDetrisPredictor:
    """预测器类"""
    
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
    
    def predict(self, multimask_output: bool = True) -> tuple:
        if not self.is_image_set:
            raise RuntimeError("Image must be set before prediction")
        
        result = self.model.forward_single(
            image=self.image,
            original_size=self.original_size,
            text_prompt=self.text_prompt,
            multimask_output=multimask_output
        )
        
        masks = result['masks']
        scores = result['iou_predictions']
        logits = result['low_res_logits']
        return masks, scores, logits
    
    def reset_image(self):
        self.image = None
        self.text_prompt = ""
        self.original_size = None
        self.is_image_set = False