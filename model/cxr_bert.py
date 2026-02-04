import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Optional
import os
from .adapter import TextAdapter
from .EMA import EMA

class FrequencyLowPassBranch(nn.Module):
    def __init__(self, input_dim, output_dim, cutoff_ratio=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cutoff_ratio = cutoff_ratio

        self.freq_weights = nn.Parameter(torch.ones(input_dim) * 0.5)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freq_dim = x_fft.size(1)
        cutoff_idx = int(freq_dim * self.cutoff_ratio)

        filter_mask = torch.ones(freq_dim, device=x.device, dtype=x.dtype)
        if cutoff_idx < freq_dim:
            filter_mask[cutoff_idx:] = torch.linspace(1.0, 0.0, freq_dim - cutoff_idx, device=x.device)
        
        learnable_weights = torch.sigmoid(self.freq_weights[:freq_dim])
        x_fft_filtered = x_fft * (filter_mask * learnable_weights).unsqueeze(0)
        x_filtered = torch.fft.irfft(x_fft_filtered, n=self.input_dim, dim=1, norm='ortho')

        return self.proj(x_filtered)


class FrequencyHighPassBranch(nn.Module):
    def __init__(self, input_dim, output_dim, cutoff_ratio=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cutoff_ratio = cutoff_ratio

        self.freq_weights = nn.Parameter(torch.ones(input_dim) * 0.5)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        freq_dim = x_fft.size(1)
        cutoff_idx = int(freq_dim * self.cutoff_ratio)

        filter_mask = torch.zeros(freq_dim, device=x.device, dtype=x.dtype)
        if cutoff_idx < freq_dim:
            filter_mask[cutoff_idx:] = torch.linspace(0.0, 1.0, freq_dim - cutoff_idx, device=x.device)
        
        learnable_weights = torch.sigmoid(self.freq_weights[:freq_dim])
        x_fft_filtered = x_fft * (filter_mask * learnable_weights).unsqueeze(0)
        x_filtered = torch.fft.irfft(x_fft_filtered, n=self.input_dim, dim=1, norm='ortho')

        return self.proj(x_filtered)


class MedicalMetaNet(nn.Module):
    def __init__(self, visual_dim: int = 768, hidden_dim: int = 48, 
                 output_dim: int = 768, n_visual_tokens: int = 9):
        super().__init__()

        if hidden_dim < 256:
            actual_hidden_dim = 256
        else:
            actual_hidden_dim = hidden_dim
            
        if actual_hidden_dim % 2 != 0:
            actual_hidden_dim = ((actual_hidden_dim // 2) + 1) * 2

        sqrt_tokens = int(n_visual_tokens ** 0.5)
        if sqrt_tokens * sqrt_tokens != n_visual_tokens:
            n_visual_tokens = 9
            sqrt_tokens = 3
        
        self.n_visual_tokens = n_visual_tokens
        self.sqrt_tokens = sqrt_tokens

        self.visual_token_expander = nn.Sequential(
            nn.Linear(visual_dim, actual_hidden_dim * n_visual_tokens),
            nn.LayerNorm(actual_hidden_dim * n_visual_tokens),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.visual_pos_embedding = nn.Parameter(
            torch.randn(n_visual_tokens, actual_hidden_dim) * 0.02
        )

        self.feature_transform = nn.Sequential(
            nn.LayerNorm(actual_hidden_dim),
            nn.Linear(actual_hidden_dim, actual_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(actual_hidden_dim * 2, actual_hidden_dim)
        )

        self.medical_ema = EMA(
            channels=actual_hidden_dim,
            factor=2
        )

        self.anatomy_enhancer = FrequencyLowPassBranch(
            input_dim=actual_hidden_dim // 2,
            output_dim=actual_hidden_dim,
            cutoff_ratio=0.3
        )

        self.pathology_enhancer = FrequencyHighPassBranch(
            input_dim=actual_hidden_dim // 2,
            output_dim=actual_hidden_dim,
            cutoff_ratio=0.3
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(actual_hidden_dim * 2),
            nn.Linear(actual_hidden_dim * 2, output_dim),
            nn.Tanh()
        )
        
        self.visual_dim = visual_dim
        self.hidden_dim = actual_hidden_dim
        self.output_dim = output_dim
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:

        batch_size = visual_features.size(0)
        
        # Step 1:
        tokens = self.visual_token_expander(visual_features)
        tokens = tokens.view(batch_size, self.n_visual_tokens, self.hidden_dim)
        tokens = tokens + self.visual_pos_embedding.unsqueeze(0)
        
        # Step 2:
        tokens = tokens + self.feature_transform(tokens)
        
        # Step 3:
        features_2d = tokens.transpose(1, 2).reshape(
            batch_size, self.hidden_dim, self.sqrt_tokens, self.sqrt_tokens
        )
        
        # Step 4:
        ema_features = self.medical_ema(features_2d)
        
        # Step 5:
        anatomy_2d = ema_features[:, :self.hidden_dim//2, :, :]
        pathology_2d = ema_features[:, self.hidden_dim//2:, :, :]
        
        anatomy_input = anatomy_2d.mean(dim=[2, 3])
        pathology_input = pathology_2d.max(dim=2)[0].max(dim=2)[0]
        
        # Step 6:
        enhanced_anatomy = self.anatomy_enhancer(anatomy_input)
        
        enhanced_pathology = self.pathology_enhancer(pathology_input)
        
        combined = torch.cat([enhanced_anatomy, enhanced_pathology], dim=1)  # [B, 512]
        conditional_token = self.output_proj(combined)  # [B, 768]
        
        return conditional_token


class CXRBERTTextEncoder(nn.Module):
    def __init__(
        self, 
        model_name: str = "./BiomedVLP-CXR-BERT-specialized",
        context_length: int = 77,
        embed_dim: int = 512,
        add_adapter_layer: list = [],
        txt_adapter_dim: int = 384,
        use_cocoop: bool = False,
        meta_net_hidden_dim: int = 48,
        conditional_weight: float = 1.0,
        n_ctx: int = 4,
        adapter_scale: float = 0.2,
    ):
        super().__init__()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                local_files_only=True
            )
            self.bert_model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e:
            print(f"Local loading failed: {e}")
            print("Attempting online loading...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized", 
                trust_remote_code=True
            )
            self.bert_model = AutoModel.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized", 
                trust_remote_code=True
            )
        
        self.bert_hidden_dim = getattr(self.bert_model.config, 'hidden_size', 768)
        self.max_position_embeddings = getattr(self.bert_model.config, 'max_position_embeddings', 512)
        
        self.embeddings = self._get_embeddings()

        self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        self.prompt_token_id = self._get_safe_prompt_token_id()

        self._freeze_bert_weights()

        self.bert_projection = nn.Linear(self.bert_hidden_dim, embed_dim)

        self.transformer = CXRBERTTransformerWrapper(
            add_adapter_layer=add_adapter_layer,
            txt_adapter_dim=txt_adapter_dim,
            bert_model=self.bert_model,
            hidden_dim=self.bert_hidden_dim,
            adapter_scale=adapter_scale
        )

        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim) * (embed_dim ** -0.5))

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.use_cocoop = use_cocoop
        self.conditional_weight = conditional_weight
        self.n_ctx = n_ctx

        self.ctx_embeddings = nn.Parameter(torch.randn(n_ctx, self.bert_hidden_dim) * 0.02)

        if self.use_cocoop:
            self.meta_net = MedicalMetaNet(
                visual_dim=768,
                hidden_dim=meta_net_hidden_dim,
                output_dim=self.bert_hidden_dim
            )
            self.condition_projector = nn.Sequential(
                nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim * n_ctx),
                nn.GELU(),
                nn.Linear(self.bert_hidden_dim * n_ctx, self.bert_hidden_dim * n_ctx)
            )
            print(f"CoCoOp enabled: {n_ctx} learnable contexts + Meta-Net")
        else:
            self.meta_net = None
            self.condition_projector = None
            print(f"CoOp mode: {n_ctx} static learnable contexts")

        self._initialize_weights()

        self.token_embedding = self.embeddings.word_embeddings
        max_pos = self.max_position_embeddings
        self.positional_embedding = nn.Parameter(
            self.embeddings.position_embeddings.weight.data.clone()
        ) 

    def _get_embeddings(self):
        if hasattr(self.bert_model, 'embeddings'):
            return self.bert_model.embeddings
        elif hasattr(self.bert_model, 'bert') and hasattr(self.bert_model.bert, 'embeddings'):
            return self.bert_model.bert.embeddings
        else:
            raise ValueError(f"Cannot find embeddings in CXR-BERT model structure. "
                           f"Available attributes: {list(self.bert_model.__dict__.keys())}")

    def _get_safe_prompt_token_id(self):
        unused_tokens = ['[unused1]', '[unused2]', '[unused3]', '<unused1>', '<unused2>', '<unused3>']
        for token in unused_tokens:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id and token_id is not None:
                    print(f"Using {token} (id={token_id}) for prompt tokens")
                    return token_id
            except:
                continue

        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            if self.tokenizer.mask_token_id != self.pad_token_id:
                print(f"Using mask token (id={self.tokenizer.mask_token_id}) for prompt tokens")
                return self.tokenizer.mask_token_id

        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
            if self.tokenizer.unk_token_id != self.pad_token_id:
                print(f"Using unk token (id={self.tokenizer.unk_token_id}) for prompt tokens")
                return self.tokenizer.unk_token_id

        safe_id = 1
        if safe_id != self.pad_token_id:
            print(f"Using safe token id={safe_id} for prompt tokens")
            return safe_id
        
        print(f"Warning: Using pad_token_id + 1 = {self.pad_token_id + 1} for prompt tokens")
        return self.pad_token_id + 1
    
    def _freeze_bert_weights(self):

        for name, param in self.bert_model.named_parameters():
            if 'embeddings' not in name:
                param.requires_grad = False
        
        if hasattr(self.embeddings, 'word_embeddings'):
            for param in self.embeddings.word_embeddings.parameters():
                param.requires_grad = False

    def _initialize_weights(self):
        nn.init.normal_(self.ctx_embeddings, std=0.02)
        nn.init.normal_(self.bert_projection.weight, std=0.02)
        nn.init.zeros_(self.bert_projection.bias)
        
        if self.condition_projector is not None:
            for m in self.condition_projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def get_prompt_embeddings(self, batch_size: int, visual_condition: Optional[torch.Tensor] = None):
        prompt_embeds = self.ctx_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.use_cocoop and visual_condition is not None:
            condition_features = self.meta_net(visual_condition)  # [B, bert_hidden_dim]
            
            condition_bias = self.condition_projector(condition_features)  # [B, bert_hidden_dim * n_ctx]
            condition_bias = condition_bias.view(batch_size, self.n_ctx, self.bert_hidden_dim)  # [B, n_ctx, bert_hidden_dim]
            
            prompt_embeds = prompt_embeds + self.conditional_weight * condition_bias
        
        return prompt_embeds

    def encode_text(self, text: torch.Tensor, visual_condition: Optional[torch.Tensor] = None):
        batch_size, seq_len = text.shape
        device = text.device
        
        total_seq_len = seq_len + self.n_ctx
        if total_seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds maximum {self.max_position_embeddings}. "
                f"Consider reducing context length or input sequence length."
            )

        prompt_embeddings = self.get_prompt_embeddings(batch_size, visual_condition)  # [B, n_ctx, 768]

        text_word_embeddings = self.embeddings.word_embeddings(text)  # [B, seq_len, 768]

        cls_embedding = text_word_embeddings[:, 0:1, :]     # [B, 1, 768]
        text_tokens_embedding = text_word_embeddings[:, 1:, :]  # [B, seq_len-1, 768]
        
        full_word_embeddings = torch.cat([
            cls_embedding,           # [B, 1, 768]
            prompt_embeddings,       # [B, n_ctx, 768]
            text_tokens_embedding    # [B, seq_len-1, 768]
        ], dim=1)  # [B, 1+n_ctx+seq_len-1, 768]

        cls_token_id = text[:, 0:1]  # [B, 1]
        prompt_token_ids = torch.full(
            (batch_size, self.n_ctx), 
            self.prompt_token_id, 
            dtype=torch.long, 
            device=device
        )  # [B, n_ctx]
        text_token_ids = text[:, 1:]  # [B, seq_len-1]
        
        full_token_ids = torch.cat([
            cls_token_id,           # [B, 1]
            prompt_token_ids,       # [B, n_ctx] 
            text_token_ids          # [B, seq_len-1]
        ], dim=1)  # [B, 1+n_ctx+seq_len-1]

        full_seq_len = full_token_ids.size(1)
        position_ids = torch.arange(full_seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [B, full_seq_len]

        token_type_ids = torch.zeros_like(full_token_ids)

        original_attention_mask = (text != self.pad_token_id).long()
        prompt_attention_mask = torch.ones(batch_size, self.n_ctx, dtype=torch.long, device=device)
        full_attention_mask = torch.cat([
            original_attention_mask[:, 0:1],  # CLS mask
            prompt_attention_mask,            # prompt mask  
            original_attention_mask[:, 1:]    # text mask
        ], dim=1)

        try:
            bert_outputs = self.bert_model(
                input_ids=None,  
                attention_mask=full_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=full_word_embeddings,
                return_dict=True
            )
            sequence_output = bert_outputs.last_hidden_state  # [B, full_seq_len, 768]
        except Exception as e:
            print(f"BERT forward failed with error: {e}")
            print(f"Input shapes - embeddings: {full_word_embeddings.shape}, "
                  f"attention_mask: {full_attention_mask.shape}, "
                  f"token_type_ids: {token_type_ids.shape}")
            raise RuntimeError(f"BERT encoding failed: {e}. Please check input format and sequence length.")

        x = sequence_output.permute(1, 0, 2)  # [full_seq_len, B, 768]
        for layer in self.transformer.resblocks:
            x = layer(x)
        x = x.permute(1, 0, 2)  # [B, full_seq_len, 768]

        x = self.bert_projection(x)  # [B, full_seq_len, embed_dim]

        x = self.ln_final(x)
        
        cls_features = x[:, 0, :]  # [B, embed_dim]
        sentence_embedding = cls_features @ self.text_projection  # [B, embed_dim]

        sentence_embedding = F.normalize(sentence_embedding, dim=-1)
        
        return x, sentence_embedding

    def set_debug_mode(self, debug: bool = True):
        self._debug_cocoop = debug
    
    @property
    def dtype(self):
        return self.text_projection.dtype


class CXRBERTTransformerWrapper(nn.Module):
    def __init__(self, add_adapter_layer=[], txt_adapter_dim=384, bert_model=None, hidden_dim=768, adapter_scale=0.2):
        super().__init__()
        
        self.adapter_scale = nn.Parameter(torch.tensor(adapter_scale))  # 改进4: 可学习的scale
        
        adapter_dict = {}
        for layer_idx in add_adapter_layer:
            adapter_dict[str(layer_idx)] = TextAdapter(
                fc_in_channels=hidden_dim,  # 使用实际的hidden_dim
                in_channels=txt_adapter_dim,
                ch1x1=txt_adapter_dim // 2,
                ch3x3red=txt_adapter_dim // 16,
                ch3x3=txt_adapter_dim // 4,
                ch5x5red=txt_adapter_dim // 16,
                ch5x5=txt_adapter_dim // 4,
                skip_connect=False
            )

        self.adapters = nn.ModuleDict(adapter_dict)
        
        num_layers = getattr(bert_model.config if bert_model else None, 'num_hidden_layers', 12)
        self.resblocks = nn.ModuleList([
            SimpleAdapterLayer(adapter_dict.get(str(i), None), self.adapter_scale)
            for i in range(num_layers)
        ])


class SimpleAdapterLayer(nn.Module):
    def __init__(self, adapter=None, adapter_scale=None):
        super().__init__()
        self.adapter = adapter
        self.adapter_scale = adapter_scale
        
    def forward(self, x):
        """x: [seq_len, batch_size, hidden_dim]"""
        if self.adapter is not None:
            x_bld = x.permute(1, 0, 2)
            adapter_output = self.adapter(x_bld)
            scale = self.adapter_scale if self.adapter_scale is not None else 0.2
            x_bld = x_bld + scale * adapter_output
            x = x_bld.permute(1, 0, 2)
        return x


def build_cxr_bert_model(
    model_name: str = "/root/DETRIS_SAM/model/BiomedVLP-CXR-BERT-specialized",
    txt_length: int = 77,
    new_resolution: int = -1,
    add_adapter_layer: list = [],
    txt_adapter_dim: int = 384,
    use_cocoop: bool = False,
    meta_net_hidden_dim: int = 48,
    conditional_weight: float = 1.0,
    n_ctx: int = 4,
    adapter_scale: float = 0.2
) -> CXRBERTTextEncoder:

    return CXRBERTTextEncoder(
        model_name=model_name,
        context_length=txt_length,
        embed_dim=512,
        add_adapter_layer=add_adapter_layer,
        txt_adapter_dim=txt_adapter_dim,
        use_cocoop=use_cocoop,
        meta_net_hidden_dim=meta_net_hidden_dim,
        conditional_weight=conditional_weight,
        n_ctx=n_ctx,
        adapter_scale=adapter_scale
    )