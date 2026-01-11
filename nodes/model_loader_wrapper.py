"""
DreamID-V Model Loader Wrapper
Direct path loading without temporary directories
"""

import torch
import os
import folder_paths
import logging
from easydict import EasyDict
import copy

# Import original components
from ..dreamidv_wan.modules.model import WanModel
from ..dreamidv_wan.modules.vae import WanVAE
from ..dreamidv_wan.modules.t5 import T5EncoderModel
from ..dreamidv_wan.configs import WAN_CONFIGS


class DreamIDV_ModelLoader_Wrapper_TTP:
    """
    Loads from separate directories WITHOUT temp files
    Uses caching to avoid reloading models when paths haven't changed
    """
    
    # Class-level cache for models
    _cached_wrapper = None
    _cached_paths = None
    
    @classmethod
    def INPUT_TYPES(cls):
        dreamidv_models = folder_paths.get_filename_list("diffusion_models")
        t5_files = folder_paths.get_filename_list("text_encoders")
        
        return {
            "required": {
                "dreamidv_model": (dreamidv_models, {"default": "dreamidv.pth"}),
                "t5_checkpoint": (t5_files, {"default": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16", "tooltip": "Precision for the Diffusion Model. bf16/fp16 recommended for speed/memory."}),
                "quantization": (["disabled", "fp8_e4m3fn", "fp8_e5m2"], {"default": "disabled", "tooltip": "Experimental FP8 quantization for Diffusion Model weights."}),
            },
            "optional": {
                "attention_mode": (["auto", "flash_attn", "sageattn", "sdpa"], 
                                   {"default": "auto", "tooltip": "Attention backend: auto=best available, flash_attn=FA2/3, sageattn=SageAttention, sdpa=PyTorch"}),
            }
        }
    
    RETURN_TYPES = ("DREAMIDV_WRAPPER",)
    RETURN_NAMES = ("dreamidv_wrapper",)
    FUNCTION = "load_models"
    CATEGORY = "DreamID-V/Wrapper"
    
    def load_models(self, dreamidv_model, t5_checkpoint, precision="bf16", quantization="disabled",
                    attention_mode="sdpa"):
        """
        Direct loading with absolute paths (no VAE - use ComfyUI VAE loader)
        Uses caching to avoid reloading when paths are unchanged
        """
        # Get full paths
        dreamidv_path = folder_paths.get_full_path("diffusion_models", dreamidv_model)
        t5_path = folder_paths.get_full_path("text_encoders", t5_checkpoint)
        
        current_paths = (dreamidv_path, t5_path, precision, quantization, attention_mode)
        
        # Check cache - return cached wrapper if paths match
        if (DreamIDV_ModelLoader_Wrapper_TTP._cached_wrapper is not None and
            DreamIDV_ModelLoader_Wrapper_TTP._cached_paths == current_paths):
            logging.info(f"[DreamID-V Wrapper] ✓ Using cached models (no reload needed)")
            return (DreamIDV_ModelLoader_Wrapper_TTP._cached_wrapper,)
        
        # Get built-in tokenizer path
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        tokenizer_dir = os.path.join(plugin_dir, 'tokenizer_configs')
        
        logging.info(f"[DreamID-V Wrapper] Loading models (T5 + WanModel, no VAE)...")
        logging.info(f"  DreamID-V: {dreamidv_path}")
        logging.info(f"  T5: {t5_path}")
        logging.info(f"  Tokenizer (built-in): {tokenizer_dir}")
        logging.info(f"  (VAE: use ComfyUI VAE loader separately)")
        logging.info(f"  Settings: Precision={precision}, Quantization={quantization}")

        cfg = WAN_CONFIGS['swapface']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine target dtypes
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        base_dtype = dtype_map.get(precision, torch.bfloat16)

        quant_dtype = None
        if quantization == "fp8_e4m3fn":
            quant_dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            quant_dtype = torch.float8_e5m2

        # Create wrapper matching original DreamIDV structure
        class DreamIDVWrapper:
            pass
        
        wrapper = DreamIDVWrapper()
        wrapper.device = device
        wrapper.config = cfg
        wrapper.rank = 0
        wrapper.t5_cpu = False
        wrapper.num_train_timesteps = cfg.num_train_timesteps
        wrapper.param_dtype = base_dtype
        wrapper.vae_stride = cfg.vae_stride
        wrapper.patch_size = cfg.patch_size
        wrapper.sp_size = 1
        
        # Load T5 with format detection
        logging.info(f"[DreamID-V Wrapper] Loading T5...")
        
        # Load state dict to detect format
        if t5_checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file
            t5_state_dict = load_file(t5_path)
        else:
            t5_state_dict = torch.load(t5_path, map_location='cpu')
        
        # Detect format by checking key names
        is_hf_format = any('encoder.block' in k for k in t5_state_dict.keys())
        
        if is_hf_format:
            logging.info(f"[DreamID-V Wrapper] Detected HuggingFace T5 format...")
            
            # Auto-detect FP8 quantization (learned from WanVideoWrapper)
            quantization = "disabled"
            is_scaled_fp8 = "scaled_fp8" in t5_state_dict
            
            for k, v in t5_state_dict.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float8_e4m3fn:
                    quantization = "fp8_e4m3fn_scaled" if is_scaled_fp8 else "fp8_e4m3fn"
                    logging.info(f"[DreamID-V Wrapper] Detected FP8 quantization (scaled={is_scaled_fp8})")
                    break
            
            # Extract scale_weights for scaled FP8 (WanVideoWrapper approach)
            scale_weights = {}
            if is_scaled_fp8 or "fp8" in quantization:
                for k, v in t5_state_dict.items():
                    if k.endswith(".scale_weight"):
                        scale_weights[k] = v.to('cpu', cfg.t5_dtype)
                        logging.info(f"[DreamID-V Wrapper] Extracted scale_weight: {k}")
            
            # Convert HuggingFace format to DreamID-V format (concise logging)
            logging.info(f"[DreamID-V Wrapper] Converting HuggingFace format to DreamID-V format...")
            converted_sd = {}
            
            convert_count = 0
            for key, value in t5_state_dict.items():
                # Skip scale_weight keys, they'll be handled separately
                if key.endswith(".scale_weight"):
                    continue
                    
                if key.startswith('encoder.block.'):
                    parts = key.split('.')
                    block_num = parts[2]
                    
                    # Self-attention components
                    if 'layer.0.SelfAttention' in key:
                        if key.endswith('.k.weight'):
                            new_key = f"blocks.{block_num}.attn.k.weight"
                        elif key.endswith('.o.weight'):
                            new_key = f"blocks.{block_num}.attn.o.weight"
                        elif key.endswith('.q.weight'):
                            new_key = f"blocks.{block_num}.attn.q.weight"
                        elif key.endswith('.v.weight'):
                            new_key = f"blocks.{block_num}.attn.v.weight"
                        elif 'relative_attention_bias' in key:
                            new_key = f"blocks.{block_num}.pos_embedding.embedding.weight"
                        else:
                            new_key = key
                    
                    # Layer norms
                    elif 'layer.0.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm1.weight"
                    elif 'layer.1.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm2.weight"
                    
                    # Feed-forward components
                    elif 'layer.1.DenseReluDense' in key:
                        if 'wi_0' in key:
                            new_key = f"blocks.{block_num}.ffn.gate.0.weight"
                        elif 'wi_1' in key:
                            new_key = f"blocks.{block_num}.ffn.fc1.weight"
                        elif 'wo' in key:
                            new_key = f"blocks.{block_num}.ffn.fc2.weight"
                        else:
                            new_key = key
                    else:
                        new_key = key
                
                elif key == "shared.weight":
                    new_key = "token_embedding.weight"
                elif key == "encoder.final_layer_norm.weight":
                    new_key = "norm.weight"
                else:
                    new_key = key
                
                converted_sd[new_key] = value
                
                # Also convert corresponding scale_weight key if exists
                scale_key = f"{key}.scale_weight"
                if scale_key in scale_weights:
                    converted_scale_key = f"{new_key}.scale_weight"
                    scale_weights[converted_scale_key] = scale_weights.pop(scale_key)
            
            # Load model with converted state dict
            from ..dreamidv_wan.modules.t5 import umt5_xxl
            from ..dreamidv_wan.modules.tokenizers import HuggingfaceTokenizer
            
            t5_model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=cfg.t5_dtype,
                device=torch.device('cpu')
            ).eval().requires_grad_(False)
            
            # Load converted weights (may have missing keys for pos_embedding, that's OK)
            load_result = t5_model.load_state_dict(converted_sd, strict=False)
            if load_result.missing_keys:
                logging.info(f"[DreamID-V Wrapper] Missing keys (expected): {load_result.missing_keys}")
            
            # Apply FP8 optimization for scaled FP8 (WanVideoWrapper's fp8_optimization.py approach)
            if is_scaled_fp8 and len(scale_weights) > 0:
                logging.info(f"[DreamID-V Wrapper] Applying FP8 linear optimization with {len(scale_weights)} scale weights...")
                
                # Inline fp8_linear_forward (from WanVideoWrapper's fp8_optimization.py)
                def fp8_linear_forward(linear_module, base_dtype, input_tensor):
                    weight_dtype = linear_module.weight.dtype
                    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        if len(input_tensor.shape) == 3:
                            input_shape = input_tensor.shape
                            
                            scale_weight = getattr(linear_module, 'scale_weight', None)
                            if scale_weight is None:
                                scale_weight = torch.ones((), device=input_tensor.device, dtype=torch.float32)
                            else:
                                scale_weight = scale_weight.to(input_tensor.device).squeeze()
                            
                            scale_input = torch.ones((), device=input_tensor.device, dtype=torch.float32)
                            input_tensor = torch.clamp(input_tensor, min=-448, max=448, out=input_tensor)
                            inn = input_tensor.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
                            
                            bias = linear_module.bias.to(base_dtype) if linear_module.bias is not None else None
                            o = torch._scaled_mm(inn, linear_module.weight.t(), out_dtype=base_dtype, bias=bias, 
                                               scale_a=scale_input, scale_b=scale_weight)
                            
                            return o.reshape((-1, input_shape[1], linear_module.weight.shape[0]))
                        else:
                            return linear_module.original_forward(input_tensor.to(base_dtype))
                    else:
                        return linear_module.original_forward(input_tensor)
                
                # Apply to all Linear layers
                import torch.nn as nn
                params_to_keep = {"norm", "bias", "embedding"}
                
                for name, module in t5_model.named_modules():
                    if not any(keyword in name for keyword in params_to_keep):
                        if isinstance(module, nn.Linear):
                            scale_key = f"{name}.scale_weight"
                            if scale_key in scale_weights:
                                setattr(module, "scale_weight", scale_weights[scale_key].float())
                            original_forward = module.forward
                            setattr(module, "original_forward", original_forward)
                            setattr(module, "forward", lambda inp, m=module: fp8_linear_forward(m, cfg.t5_dtype, inp))
                
                logging.info(f"[DreamID-V Wrapper] ✓ FP8 optimization applied")
            
            t5_model.to(torch.device('cpu'))
            
            # Use sentencepiece tokenizer for HF T5 (UMT5 is based on sentencepiece)
            import sentencepiece as spm
            
            spiece_model_path = os.path.join(tokenizer_dir, 'spiece.model')
            if not os.path.exists(spiece_model_path):
                logging.info(f"[DreamID-V Wrapper] Downloading spiece.model for HF T5...")
                try:
                    import urllib.request
                    url = "https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model"
                    urllib.request.urlretrieve(url, spiece_model_path)
                    logging.info(f"[DreamID-V Wrapper] ✓ Downloaded spiece.model")
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download spiece.model: {e}\n"
                        f"Please manually download from: https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model\n"
                        f"And place it in: {tokenizer_dir}"
                    )
            
            sp = spm.SentencePieceProcessor()
            sp.load(spiece_model_path)
            
            class T5EncoderWrapper:
                def __init__(self, model, sp_processor):
                    self.model = model
                    self.sp = sp_processor
                    self.text_len = cfg.text_len
                    self.dtype = cfg.t5_dtype
                    self.device = torch.device('cpu')
                    self.checkpoint_path = t5_path
                    self.tokenizer_path = tokenizer_dir
                
                def __call__(self, texts, device):
                    # Tokenize with sentencepiece
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    ids_list = []
                    for text in texts:
                        ids = self.sp.encode(text, out_type=int)
                        # Pad or truncate to text_len
                        if len(ids) > self.text_len:
                            ids = ids[:self.text_len]
                        else:
                            ids = ids + [self.sp.pad_id()] * (self.text_len - len(ids))
                        ids_list.append(ids)
                    
                    ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=device)
                    mask = (ids_tensor != self.sp.pad_id()).long()
                    seq_lens = mask.sum(dim=1).long()
                    
                    context = self.model(ids_tensor, mask)
                    return [u[:v] for u, v in zip(context, seq_lens)]
            
            wrapper.text_encoder = T5EncoderWrapper(t5_model, sp)
            logging.info(f"[DreamID-V Wrapper] ✓ T5 loaded (HuggingFace format converted, quantization={quantization})")
            
        else:
            logging.info(f"[DreamID-V Wrapper] Detected original DreamID-V T5 format...")
            
            # Use original loading for DreamID-V format
            from ..dreamidv_wan.modules.t5 import umt5_xxl
            import sentencepiece as spm
            
            t5_model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=cfg.t5_dtype,
                device=torch.device('cpu')
            ).eval().requires_grad_(False)
            
            t5_model.load_state_dict(t5_state_dict)
            t5_model.to(torch.device('cpu'))
            
            # Load sentencepiece model
            spiece_model_path = os.path.join(tokenizer_dir, 'spiece.model')
            
            # Check if file exists and is valid
            needs_download = True
            if os.path.exists(spiece_model_path):
                try:
                    # Try to validate the file
                    test_sp = spm.SentencePieceProcessor()
                    test_sp.load(spiece_model_path)
                    needs_download = False
                    logging.info(f"[DreamID-V Wrapper] Using existing spiece.model")
                except:
                    logging.warning(f"[DreamID-V Wrapper] Existing spiece.model is corrupted, re-downloading...")
                    os.remove(spiece_model_path)
            
            if needs_download:
                logging.info(f"[DreamID-V Wrapper] Downloading spiece.model...")
                try:
                    import urllib.request
                    url = "https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model"
                    urllib.request.urlretrieve(url, spiece_model_path)
                    logging.info(f"[DreamID-V Wrapper] ✓ Downloaded spiece.model")
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download spiece.model: {e}\n"
                        f"Please manually download from:\n"
                        f"https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model\n"
                        f"And place it in: {tokenizer_dir}"
                    )
            
            # Create sentencepiece processor
            sp = spm.SentencePieceProcessor()
            sp.load(spiece_model_path)
            
            class T5EncoderWrapper:
                def __init__(self, model, sp_processor):
                    self.model = model
                    self.sp = sp_processor
                    self.text_len = cfg.text_len
                    self.dtype = cfg.t5_dtype
                    self.device = torch.device('cpu')
                    self.checkpoint_path = t5_path
                    self.tokenizer_path = tokenizer_dir
                
                def __call__(self, texts, device):
                    # Tokenize with sentencepiece
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    ids_list = []
                    for text in texts:
                        ids = self.sp.encode(text, out_type=int)
                        # Pad or truncate to text_len
                        if len(ids) > self.text_len:
                            ids = ids[:self.text_len]
                        else:
                            ids = ids + [self.sp.pad_id()] * (self.text_len - len(ids))
                        ids_list.append(ids)
                    
                    ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=device)
                    mask = (ids_tensor != self.sp.pad_id()).long()
                    seq_lens = mask.sum(dim=1).long()
                    
                    context = self.model(ids_tensor, mask)
                    return [u[:v] for u, v in zip(context, seq_lens)]
            
            wrapper.text_encoder = T5EncoderWrapper(t5_model, sp)
            logging.info(f"[DreamID-V Wrapper] ✓ T5 loaded (original format)")

        # Note: VAE is NOT loaded here - use ComfyUI's Load VAE node instead
        # The VAE is passed to sampler_wrapper for encoding reference images
        
        # Set attention mode before loading model
        from ..dreamidv_wan.modules.attention import set_attention_mode, SAGE_ATTN_AVAILABLE, FLASH_ATTN_2_AVAILABLE
        logging.info(f"[DreamID-V Wrapper] Setting attention mode: {attention_mode}")
        logging.info(f"  Available backends - Flash Attn: {FLASH_ATTN_2_AVAILABLE}, SageAttn: {SAGE_ATTN_AVAILABLE}")
        set_attention_mode(attention_mode)
        
        # Load WanModel
        logging.info(f"[DreamID-V Wrapper] Loading WanModel...")
        logging.info(f"  - Base Precision: {precision}")
        logging.info(f"  - Quantization: {quantization} (Note: FP8 requires model support, otherwise behaves as base_precision)")

        wrapper.model = WanModel(
            model_type=cfg.model_type,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            in_dim=cfg.in_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps
        )
        
        # Apply Mixed Precision (Reference: WanVideoWrapper)
        # 1. Cast entire model to base_dtype
        wrapper.model.to(dtype=base_dtype)
        
        # 2. Restore sensitive layers to FP32
        # Keys to keep in FP32/Higher Precision
        params_to_keep_fp32 = ["patch_embedding", "motion_encoder", "condition_embedding", "norm", "bias", "time_in", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add", "ref_conv", "audio_proj"]
        
        count_fp32 = 0
        for name, param in wrapper.model.named_parameters():
             if any(k in name for k in params_to_keep_fp32):
                  param.data = param.data.to(dtype=torch.float32)
                  count_fp32 += 1
        logging.info(f"  - Kept {count_fp32} parameters in FP32 for stability")

        # 3. Load State Dict (CPU) to avoid VRAM spike
        state = torch.load(dreamidv_path, map_location='cpu')
        
        # 4. Cast state_dict values to match model parameter dtypes
        # This handles the mixed precision loading efficiently
        model_state_dict = wrapper.model.state_dict()
        new_state = {}
        
        for k, v in state.items():
            if k in model_state_dict:
                # Cast source tensor to match the target model parameter's dtype
                # e.g. if model param is bf16, cast loaded tensor to bf16
                target_dtype = model_state_dict[k].dtype
                new_state[k] = v.to(dtype=target_dtype)
            else:
                new_state[k] = v # Keep as is if not in model (though load_state_dict might ignore it)
        
        del state # Free original memory
        
        wrapper.model.load_state_dict(new_state, strict=False)
        wrapper.model.eval().requires_grad_(False).to(device)
        logging.info(f"[DreamID-V Wrapper] ✓ WanModel loaded")
        
        # Add methods from original
        from ..dreamidv_wan.wan_swapface import DreamIDV
        wrapper.load_image_latent_ref_ip_video = DreamIDV.load_image_latent_ref_ip_video.__get__(wrapper, DreamIDVWrapper)
        wrapper.generate = DreamIDV.generate.__get__(wrapper, DreamIDVWrapper)
        
        logging.info(f"[DreamID-V Wrapper] ✓ Pipeline ready")
        
        # Update cache
        DreamIDV_ModelLoader_Wrapper_TTP._cached_wrapper = wrapper
        DreamIDV_ModelLoader_Wrapper_TTP._cached_paths = current_paths
        logging.info(f"[DreamID-V Wrapper] ✓ Models cached for future runs")
        
        return (wrapper,)


NODE_CLASS_MAPPINGS = {
    "DreamIDV_ModelLoader_Wrapper_TTP": DreamIDV_ModelLoader_Wrapper_TTP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamIDV_ModelLoader_Wrapper_TTP": "DreamID-V Model Loader (Wrapper)"
}

