import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.cxr_bert import build_cxr_bert_model
from .layers import Neck, Decoder, Projector
from .fusion import Fusion
from .dinov2.models.vision_transformer import vit_base, vit_large
from utils.loss import BCEDiceLoss

class CF2Seg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.use_cocoop = getattr(cfg, 'use_cocoop', False)
        self.meta_net_hidden_dim = getattr(cfg, 'meta_net_hidden_dim', 48)
        self.conditional_weight = getattr(cfg, 'conditional_weight', 1.0)
        self.n_ctx = getattr(cfg, 'n_ctx', 4)
        self.adapter_scale = getattr(cfg, 'adapter_scale', 0.2)
        self.cocoop_feature_layers = getattr(cfg, 'cocoop_feature_layers', [3, 6, 9])
        self.normalize_visual_condition = getattr(cfg, 'normalize_visual_condition', False)
        self.use_layer_attention = getattr(cfg, 'use_layer_attention', False)
        
        self.use_contrastive = getattr(cfg, 'use_contrastive', False)
        self.contrastive_temperature = getattr(cfg, 'contrastive_temperature', 0.2)
        self.use_dynamic_weights = getattr(cfg, 'use_dynamic_weights', True)
        self.current_epoch = 0
        self.total_epochs = getattr(cfg, 'total_epochs', 100)

        self.txt_backbone = build_cxr_bert_model(
            model_name=getattr(cfg, 'cxr_bert_model', '/model/BiomedVLP-CXR-BERT-specialized'),
            txt_length=cfg.word_len,
            add_adapter_layer=cfg.txtual_adapter_layer,  # [1,3,5,7,9,11]
            txt_adapter_dim=cfg.txt_adapter_dim,  # 64
            use_cocoop=self.use_cocoop,
            meta_net_hidden_dim=self.meta_net_hidden_dim,
            conditional_weight=self.conditional_weight,
            n_ctx=self.n_ctx,
            adapter_scale=self.adapter_scale
        ).float()
        
        self.fusion = Fusion(
            d_model=cfg.ladder_dim,  # 128
            nhead=cfg.nhead,         # 8
            dino_layers=cfg.dino_layers,  # 12
            output_dinov2=cfg.output_dinov2,  # [4, 8]
            use_cocoop=self.use_cocoop,
            cocoop_early_fusion=getattr(cfg, 'cocoop_early_fusion', False),
            cocoop_feature_layers=self.cocoop_feature_layers,
            normalize_visual_condition=self.normalize_visual_condition,
            use_layer_attention=self.use_layer_attention
        )

        from safetensors.torch import safe_open

        def load_safetensor_state_dict(path):
            state_dict = {}
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict

        state_dict = load_safetensor_state_dict(cfg.dino_pretrain)
        if cfg.dino_name=='RAD-DINO':
            self.dinov2 = vit_base(
                patch_size=14,
                num_register_tokens=4,
                img_size=518,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,  # [1,3,5,7,9,11]
                visual_adapter_dim=cfg.visual_adapter_dim,   # 128               
            )

        self.dinov2.load_state_dict(state_dict, strict=False)

        self.neck = Neck(
            in_channels=cfg.fpn_in,      # [768, 768, 768]
            out_channels=cfg.fpn_out,    # [256, 512, 1024]
            stride=cfg.stride            # [1, 1, 1]
        )
        self.decoder = Decoder(
            num_layers=cfg.num_layers,   # 3
            d_model=cfg.vis_dim,         # 512
            nhead=cfg.num_head,          # 8
            dim_ffn=cfg.dim_ffn,         # 512
            dropout=cfg.dropout,         # 0.1
            return_intermediate=cfg.intermediate  # False
        )
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)  # 512, 256, 3
        self.criterion = BCEDiceLoss(
            bce_weight=getattr(cfg, 'bce_weight', 0.5),
            dice_weight=getattr(cfg, 'dice_weight', 0.5)
        )

        self.setup_freezing_strategy()
        
        if self.use_cocoop:
            print(f"DETRIS initialized with CoCoOp: n_ctx={self.n_ctx}, layers={self.cocoop_feature_layers}")
        if self.use_contrastive:
            print(f"DETRIS with contrastive learning: temperature={self.contrastive_temperature}")

    def setup_freezing_strategy(self):

        for name, param in self.txt_backbone.named_parameters():
            keep_trainable = (
                'adapter' in name or
                'meta_net' in name or
                'ctx_embeddings' in name or
                'condition_projector' in name or
                'bert_projection' in name or
                'text_projection' in name or
                'ln_final' in name or
                'adapter_scale' in name
            )
            param.requires_grad = keep_trainable

        for name, param in self.dinov2.named_parameters():
            param.requires_grad = 'adapter' in name
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_dynamic_loss_weights(self):
        if not self.use_dynamic_weights:
            return 0.5, 0.5, 1.0
        
        progress = self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0.0
        
        bce_weight = 0.7 - 0.2 * progress
        
        dice_weight = 0.3 + 0.2 * progress
        
        contrastive_weight = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        return bce_weight, dice_weight, contrastive_weight

    def extract_mask_guided_embeddings(self, fq, mask):
        B, C, H, W = fq.shape
        
        pos_embed = self.get_positional_encoding(H, W, C).to(fq.device)
        fq_with_pos = fq + pos_embed.unsqueeze(0)

        if mask.shape[-2:] != fq.shape[-2:]:
            mask_resized = F.interpolate(mask, size=fq.shape[-2:], mode='nearest')
        else:
            mask_resized = mask

        foreground_mask = (mask_resized > 0.5).float()

        visual_embeds = []
        for i in range(B):
            fg_mask = foreground_mask[i, 0]
            if fg_mask.sum() > 0:
                fg_features = fq_with_pos[i] * fg_mask.unsqueeze(0)
                region_embed = fg_features.sum(dim=(-2, -1)) / fg_mask.sum()
            else:
                region_embed = fq_with_pos[i].mean(dim=(-2, -1))
            visual_embeds.append(region_embed)

        return torch.stack(visual_embeds, dim=0)

    def get_positional_encoding(self, height, width, channels):
        pe = torch.zeros(channels, height, width)

        pos_h = torch.arange(height).unsqueeze(1).float()
        div_term_h = torch.exp(torch.arange(0, channels//2, 2).float() * 
                              -(torch.log(torch.tensor(10000.0)) / (channels//2)))
        pe[0::4, :, :] = torch.sin(pos_h * div_term_h).unsqueeze(-1).repeat(1, 1, width)
        pe[1::4, :, :] = torch.cos(pos_h * div_term_h).unsqueeze(-1).repeat(1, 1, width)

        pos_w = torch.arange(width).unsqueeze(0).float()
        div_term_w = torch.exp(torch.arange(0, channels//2, 2).float() * 
                              -(torch.log(torch.tensor(10000.0)) / (channels//2)))
        pe[2::4, :, :] = torch.sin(pos_w * div_term_w).unsqueeze(0).repeat(height, 1, 1)
        pe[3::4, :, :] = torch.cos(pos_w * div_term_w).unsqueeze(0).repeat(height, 1, 1)
        
        return pe

    def nt_xent_loss(self, visual_embeds, text_embeds):
        batch_size = visual_embeds.shape[0]
        
        visual_embeds = F.normalize(visual_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)
        
        similarity_matrix = torch.mm(visual_embeds, text_embeds.t()) / self.contrastive_temperature

        labels = torch.arange(batch_size, device=visual_embeds.device)

        loss_v2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2v = F.cross_entropy(similarity_matrix.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2

    def forward(self, img, word, mask=None):
        batch_size = img.shape[0]
        
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        vis, word_features, state = self.fusion(img, word, self.txt_backbone, self.dinov2)

        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word_features, pad_mask)
        fq = fq.reshape(b, c, h, w)

        pred = self.proj(fq, state)

        if self.training:
            if mask is None:
                return pred.detach()
                
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()

            bce_weight, dice_weight, contrastive_weight = self.get_dynamic_loss_weights()
            
            bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

            pred_sigmoid = torch.sigmoid(pred)
            batch_size = pred_sigmoid.size(0)
            pred_flat = pred_sigmoid.view(batch_size, -1)
            mask_flat = mask.view(batch_size, -1)
            
            intersection = (pred_flat * mask_flat).sum(dim=1)
            pred_sum = pred_flat.sum(dim=1)
            mask_sum = mask_flat.sum(dim=1)
            
            dice_coeff = (2 * intersection + 1e-5) / (pred_sum + mask_sum + 1e-5)
            dice_loss = 1 - dice_coeff.mean()

            main_loss = bce_weight * bce_loss + dice_weight * dice_loss

            contrastive_loss_value = 0.0
            if self.use_contrastive:
                try:
                    visual_embeds = self.extract_mask_guided_embeddings(fq, mask)
                    text_embeds = state
                    contrastive_loss_value = self.nt_xent_loss(visual_embeds, text_embeds)
                except:
                    contrastive_loss_value = torch.tensor(0.0, device=pred.device)

            loss = main_loss + contrastive_weight * contrastive_loss_value

            return pred.detach(), mask, loss
        else:
            return pred.detach()
    
    def update_epoch(self, epoch, total_epochs=None):
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def set_conditional_weight(self, weight: float):
        if self.use_cocoop and hasattr(self.txt_backbone, 'conditional_weight'):
            self.txt_backbone.conditional_weight = weight
            self.conditional_weight = weight
            print(f"Conditional weight updated to: {weight}")
    
    @torch.no_grad()
    def extract_visual_features_for_analysis(self, img):
        if not self.use_cocoop:
            return None
            
        visual_condition = self.fusion.extract_visual_condition(self.dinov2, img)
        
        if hasattr(self.txt_backbone, 'meta_net') and self.txt_backbone.meta_net is not None:
            conditional_token = self.txt_backbone.meta_net(visual_condition)
        else:
            conditional_token = None
            
        return {
            "visual_condition": visual_condition,
            "conditional_token": conditional_token,
            "visual_condition_norm": visual_condition.norm(dim=-1).mean().item(),
            "conditional_token_norm": conditional_token.norm(dim=-1).mean().item() if conditional_token is not None else 0
        }