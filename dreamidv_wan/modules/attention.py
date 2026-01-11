# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

import warnings

# Global setting for attention mode
# Can be: 'auto', 'flash_attn', 'sageattn', 'sdpa'
ATTENTION_MODE = 'auto'

def set_attention_mode(mode):
    """Set global attention mode. Options: 'auto', 'flash_attn', 'sageattn', 'sdpa'"""
    global ATTENTION_MODE
    valid_modes = ['auto', 'flash_attn', 'sageattn', 'sdpa']
    if mode not in valid_modes:
        raise ValueError(f"Invalid attention mode '{mode}'. Must be one of: {valid_modes}")
    ATTENTION_MODE = mode
    return ATTENTION_MODE

__all__ = [
    'flash_attention',
    'attention',
    'set_attention_mode',
    'SAGE_ATTN_AVAILABLE',
    'FLASH_ATTN_2_AVAILABLE',
    'FLASH_ATTN_3_AVAILABLE',
]



def _flash_attention_forward(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Original flash_attention implementation (now internal).
    Calls FA3 or FA2 directly without checking ATTENTION_MODE global.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        if not FLASH_ATTN_2_AVAILABLE:
             # Fallback if FA2 is requested but not available
             # This should catch cases where _flash_attention_forward is called directly but FA is missing
             warnings.warn(
                'Flash attention 2 is not available. Falling back to SDPA.'
             )
             return _sdpa_forward(q, k, v, q_lens, k_lens, dropout_p, causal, dtype)

        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def _sdpa_forward(q, k, v, q_lens=None, k_lens=None, dropout_p=0., causal=False, dtype=torch.bfloat16):
    """Internal SDPA fallback"""
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )
    attn_mask = None

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

    out = out.transpose(1, 2).contiguous()
    return out


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Dispatcher for attention. This replaces the original flash_attention implementation
    so that existing calls in model.py (which import flash_attention) are routed here.
    """
    global ATTENTION_MODE
    global _ATTENTION_LOGGED
    
    # Initialize logger flag if not exists
    if not globals().get('_ATTENTION_LOGGED'):
        _ATTENTION_LOGGED = False
    
    # Determine which backend to use based on mode
    use_flash = False
    use_sage = False
    use_sdpa = False
    backend_name = "Unknown"
    
    if ATTENTION_MODE == 'auto':
        # Auto mode: Flash > Sage > SDPA
        if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
            use_flash = True
            backend_name = "Flash Attention (Auto)"
        elif SAGE_ATTN_AVAILABLE:
            use_sage = True
            backend_name = "SageAttention (Auto)"
        else:
            use_sdpa = True
            backend_name = "SDPA (Auto)"
    elif ATTENTION_MODE == 'flash_attn':
        if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
            use_flash = True
            backend_name = "Flash Attention (Forced)"
        else:
            warnings.warn("Flash Attention not available, falling back to SDPA")
            use_sdpa = True
            backend_name = "SDPA (Fallback from Flash)"
    elif ATTENTION_MODE == 'sageattn':
        if SAGE_ATTN_AVAILABLE:
            use_sage = True
            backend_name = "SageAttention (Forced)"
        else:
            warnings.warn("SageAttention not available, falling back to SDPA")
            use_sdpa = True
            backend_name = "SDPA (Fallback from Sage)"
    else:  # 'sdpa' or unknown
        use_sdpa = True
        backend_name = "SDPA (Forced/Default)"
        
    # Log once
    if not _ATTENTION_LOGGED:
        import logging
        logging.info(f"[DreamID-V Attention] First call using backend: {backend_name}")
        _ATTENTION_LOGGED = True
    
    # Execute attention
    if use_flash:
        return _flash_attention_forward(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )
    elif use_sage:
        # SageAttention expects [B, N, L, C] format but we have [B, L, N, C]
        # Transpose to match
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using SageAttention. It can have a significant impact on performance.'
            )
        
        # [B, L, N, C] -> [B, N, L, C]
        q_sage = q.transpose(1, 2).to(dtype)
        k_sage = k.transpose(1, 2).to(dtype)
        v_sage = v.transpose(1, 2).to(dtype)
        
        # SageAttention call
        out = sageattn(q_sage, k_sage, v_sage, is_causal=causal)
        
        # [B, N, L, C] -> [B, L, N, C]
        out = out.transpose(1, 2).contiguous()
        return out
    else:
        # SDPA fallback
        return _sdpa_forward(q, k, v, q_lens, k_lens, dropout_p, causal, dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # Just redirect to our robust flash_attention dispatcher
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )

