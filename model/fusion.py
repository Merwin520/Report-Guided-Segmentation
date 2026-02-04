import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
import numpy as np
import torch.distributed as dist
from .layers import conv_layer, deconv_layer
import os
from functools import partial


class Fusion(nn.Module):
    def __init__(self,
                 d_img=[768, 768, 768],
                 d_txt=512,
                 d_model=64,
                 nhead=8,
                 num_stages=3,
                 strides=[1, 1, 1],
                 num_layers=12,
                 shared_weights=False,
                 dino_layers=12,
                 output_dinov2=[4, 8],
                 # CoCoOp parameters
                 use_cocoop=False,
                 cocoop_early_fusion=False,
                 cocoop_feature_layers=[3, 6, 9],
                 normalize_visual_condition=False,
                 use_layer_attention=False,
                 ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0
        
        # CoCoOp attributes
        self.use_cocoop = use_cocoop
        self.cocoop_early_fusion = cocoop_early_fusion
        self.cocoop_feature_layers = cocoop_feature_layers
        self.normalize_visual_condition = normalize_visual_condition
        self.use_layer_attention = use_layer_attention

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        
        self.initialize_parameters()
        
        if self.use_cocoop:
            print(f"Fusion module: CoCoOp enabled")
            print(f"  - Feature layers: {cocoop_feature_layers}")
            print(f"  - Early fusion: {cocoop_early_fusion}")
        else:
            print("Fusion module: Standard fusion without CoCoOp")

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def extract_visual_condition(self, dino, img):
        with torch.no_grad():
            temp_dino_f = dino.patch_embed(img)
            temp_dino_f = torch.cat((dino.cls_token.expand(temp_dino_f.shape[0], -1, -1), temp_dino_f), dim=1)
            
            B, nc, w, h = img.shape
            temp_dino_f = temp_dino_f + dino.interpolate_pos_encoding(temp_dino_f, w, h)
            
            temp_dino_f = torch.cat(
                (
                    temp_dino_f[:, :1],
                    dino.register_tokens.expand(temp_dino_f.shape[0], -1, -1),
                    temp_dino_f[:, 1:],
                ),
                dim=1,
            )
            
            target_layers = self.cocoop_feature_layers
            layer_features = []
            
            current_layer = 0
            for target_layer in target_layers:
                while current_layer <= target_layer and current_layer < len(dino.blocks):
                    temp_dino_f = dino.blocks[current_layer](temp_dino_f, None)
                    current_layer += 1
                
                if current_layer > target_layer:
                    cls_token = temp_dino_f[:, 0, :].clone()
                    layer_features.append(cls_token)
                    
                    if self.cocoop_early_fusion and hasattr(self, '_cocoop_debug') and self._cocoop_debug:
                        print(f"CoCoOp: Extracted layer {target_layer+1}, norm={cls_token.norm(dim=-1).mean().item():.4f}")
            
            if len(layer_features) == 0:
                visual_condition = temp_dino_f[:, 0, :]
                print("Warning: CoCoOp failed to extract multi-layer features, using current CLS token")
            elif len(layer_features) == 1:
                visual_condition = layer_features[0]
            else:
                num_layers = len(layer_features)
                weights = torch.linspace(0.1, 0.9, num_layers, device=layer_features[0].device)
                weights = weights / weights.sum()  
                
                visual_condition = torch.zeros_like(layer_features[0])
                for i, feature in enumerate(layer_features):
                    visual_condition += weights[i] * feature
            
            if self.normalize_visual_condition:
                visual_condition = F.normalize(visual_condition, p=2, dim=-1)
            
            if self.cocoop_early_fusion and hasattr(self, '_cocoop_debug') and self._cocoop_debug:
                print(f"CoCoOp: Fused feature norm={visual_condition.norm(dim=-1).mean().item():.4f}")
            
        return visual_condition

    def forward(self, img, text, txt_backbone, dino):
        B = img.shape[0]
        img = img.type(txt_backbone.dtype)
        vis_outs = []
        
        if self.use_cocoop and hasattr(txt_backbone, 'use_cocoop') and txt_backbone.use_cocoop:
            # Extract visual condition (multi-layer fusion)
            visual_condition = self.extract_visual_condition(dino, img)
            
            # CoCoOp conditioned text encoding
            if hasattr(txt_backbone, 'encode_text'):
                # Method 1: Use dedicated encode_text method (Recommended)
                txt_features, state = txt_backbone.encode_text(text, visual_condition)
                txt = txt_features
                original_length = text.shape[1]
                if txt.shape[1] > original_length:
                    txt = txt[:, :original_length, :]
            else:
                # Method 2: Manual conditioning (Fallback)
                txt = txt_backbone.token_embedding(text).type(txt_backbone.dtype)
                
                # Generate and add conditional token
                if hasattr(txt_backbone, 'meta_net') and txt_backbone.meta_net is not None:
                    conditional_token = txt_backbone.meta_net(visual_condition)
                    conditional_weight = getattr(txt_backbone, 'conditional_weight', 1.0)
                    txt = txt + conditional_weight * conditional_token.unsqueeze(1)
                
                txt_enc = txt_backbone.transformer
                txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]
                txt = txt.permute(1, 0, 2)  # BLD -> LBD
                
                for i in range(self.num_layers):
                    txt = txt_enc.resblocks[i](txt)
                
                txt = txt.permute(1, 0, 2)  # LBD -> BLD
                txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
                
                # Extract sentence-level features
                state = txt[torch.arange(txt.shape[0]),
                            text.argmax(dim=-1)] @ txt_backbone.text_projection
        else:
            txt = txt_backbone.token_embedding(text).type(txt_backbone.dtype)
            
            txt_enc = txt_backbone.transformer
            txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]
            txt = txt.permute(1, 0, 2)  # BLD -> LBD
            
            for i in range(self.num_layers):
                txt = txt_enc.resblocks[i](txt)

            txt = txt.permute(1, 0, 2)  # LBD -> BLD
            if hasattr(txt_backbone, 'bert_projection') and txt.size(-1) != txt_backbone.embed_dim:
                txt = txt_backbone.bert_projection(txt)
            
            txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
            
            state = txt[torch.arange(txt.shape[0]),
                        text.argmax(dim=-1)] @ txt_backbone.text_projection
        
        # DINO visual encoding
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat((dino.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
        dino_f = dino_f + dino.interpolate_pos_encoding(dino_f, w, h)
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(dino_f.shape[0], -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )
        
        features_dino = []
        
        # Cross-modal interaction via DINO blocks
        for i in range(self.dino_layers):
            dino_f = dino.blocks[i](dino_f, txt)
            if i in self.output_dinov2:
                features_dino.append(dino_f)
        
        dino_f = dino.norm(dino_f)
        features_dino.append(dino_f)
        
        for i, feature_dino in enumerate(features_dino):
            # Remove CLS token and register tokens
            feature_dino = feature_dino[:, 4 + 1 :]  # Remove 1 CLS + 4 registers
            B, L, C = feature_dino.shape
            H = int(L ** 0.5)
            W = L // H
            feature_dino = feature_dino.reshape(B, H, W, C).permute(0, 3, 1, 2)
            vis_outs.append(feature_dino)

        return vis_outs, txt, state

    def set_cocoop_debug(self, debug: bool = True):
        """Set CoCoOp debug mode."""
        self._cocoop_debug = debug
        if hasattr(self, '_cocoop_debug') and debug:
            print("Fusion: CoCoOp debug mode enabled")