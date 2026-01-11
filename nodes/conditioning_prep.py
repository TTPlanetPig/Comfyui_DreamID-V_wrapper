"""
DreamIDV Conditioning Prep Node
Prepares multi-modal conditioning for DreamID-V model
Injects reference image, video, pose, and mask into Wan conditioning
"""
import torch
import logging
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
from comfy_api.latest import io


class DreamIDV_ConditioningPrep_TTP(io.ComfyNode):
    """
    Prepare DreamID-V conditioning by encoding reference data and injecting into Wan conditioning
    This node bridges the gap between DreamID-V requirements and ComfyUI's Wan infrastructure
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DreamIDV_ConditioningPrep_TTP",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive text conditioning"),
                io.Conditioning.Input("negative", tooltip="Negative text conditioning"),
                io.Vae.Input("vae", tooltip="VAE model for encoding"),
                io.Image.Input("reference_image", tooltip="Reference face image (identity source)"),
                io.Image.Input("reference_video", tooltip="Reference video (motion source)"),
                io.Image.Input("pose_video", tooltip="Pose visualization from PoseExtractor"),
                io.Image.Input("mask_video", tooltip="Face mask from PoseExtractor"),
                io.Int.Input("width", default=1280, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=720, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=5, max=nodes.MAX_RESOLUTION, step=4,
                           tooltip="Video length in frames (must be 4n+1)"),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Conditioning.Output("positive", tooltip="Conditioning with DreamID-V data"),
                io.Conditioning.Output("negative", tooltip="Conditioning with DreamID-V data"),
                io.Latent.Output("latent", tooltip="Empty latent for sampling"),
            ],
        )
    
    @classmethod
    def execute(cls, positive, negative, vae, reference_image, reference_video,
               pose_video, mask_video, width, height, length, batch_size) -> io.NodeOutput:
        """
        Prepare conditioning with DreamID-V multi-modal inputs
        
        Args:
            positive/negative: Text conditioning
            vae: VAE model
            reference_image: Face reference [1, H, W, C]
            reference_video: Video reference [F, H, W, C]  
            pose_video: Pose visualization [F, H, W, C]
            mask_video: Mask visualization [F, H, W, C]
            width, height, length: Output dimensions
            batch_size: Batch size
            
        Returns:
            Modified conditioning with DreamID-V data injected
        """
        device = comfy.model_management.intermediate_device()
        
        # Ensure length is 4n+1
        length = ((length - 1) // 4) * 4 + 1
        
        # 1. Encode reference image (identity source)
        ref_img = comfy.utils.common_upscale(
            reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        logging.info(f"[DEBUG] ref_img before encode: {ref_img.shape}, range:[{ref_img.min():.3f}, {ref_img.max():.3f}]")
        ref_img_latent = vae.encode(ref_img[:, :, :, :3])
        logging.info(f"[DEBUG] ref_img_latent: {ref_img_latent.shape}, range:[{ref_img_latent.min():.3f}, {ref_img_latent.max():.3f}]")
        
        # 2. Encode reference video (motion source)
        ref_vid = comfy.utils.common_upscale(
            reference_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        # Pad if needed
        if ref_vid.shape[0] < length:
            pad_frames = length - ref_vid.shape[0]
            padding = torch.ones(
                (pad_frames, height, width, ref_vid.shape[-1]), 
                device=ref_vid.device, dtype=ref_vid.dtype
            ) * 0.5
            ref_vid = torch.cat([ref_vid, padding], dim=0)
        logging.info(f"[DEBUG] ref_vid before encode: {ref_vid.shape}, range:[{ref_vid.min():.3f}, {ref_vid.max():.3f}]")
        ref_vid_latent = vae.encode(ref_vid[:, :, :, :3])
        logging.info(f"[DEBUG] ref_vid_latent: {ref_vid_latent.shape}, range:[{ref_vid_latent.min():.3f}, {ref_vid_latent.max():.3f}]")
        
        # 3. Encode mask video
        # 官方: mask 不做 normalization! 保持 [0,1] 范围
        mask_vid = comfy.utils.common_upscale(
            mask_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        # Pad if needed
        if mask_vid.shape[0] < length:
            pad_frames = length - mask_vid.shape[0]
            padding = torch.zeros(
                (pad_frames, height, width, mask_vid.shape[-1]),
                device=mask_vid.device, dtype=mask_vid.dtype
            )
            mask_vid = torch.cat([mask_vid, padding], dim=0)
        # Mask should stay in [0,1], no normalization to [-1,1]
        # 官方代码注释掉了 Normalize(0.5, 0.5)
        logging.info(f"[DEBUG] mask_vid before encode: {mask_vid.shape}, range:[{mask_vid.min():.3f}, {mask_vid.max():.3f}]")
        mask_latent = vae.encode(mask_vid[:, :, :, :3])
        logging.info(f"[DEBUG] mask_latent: {mask_latent.shape}, range:[{mask_latent.min():.3f}, {mask_latent.max():.3f}]")
        
        # 4. Process pose video (no VAE encoding, use as direct embedding)
        # 官方: Rearrange("t c h w -> c t h w")
        pose_vid = comfy.utils.common_upscale(
            pose_video[:length].movedim(-1,1), width, height, "bilinear", "center"
        )  # Now: [T, C, H, W]
        # Pad if needed
        if pose_vid.shape[0] < length:
            pad_frames = length - pose_vid.shape[0]
            padding = torch.zeros(
                (pad_frames, pose_vid.shape[1], height, width),
                device=pose_vid.device, dtype=pose_vid.dtype
            )
            pose_vid = torch.cat([pose_vid, padding], dim=0)
        
        # Rearrange to [C, T, H, W] (官方格式)
        pose_embedding = pose_vid.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
        # 增加 batch 维度: [1, C, T, H, W]
        pose_embedding = pose_embedding.unsqueeze(0)
        
        # 5. 组合条件 - 按照官方格式！
        # 关键: video 和 mask 要沿 channel 维度拼接！
        # ref_vid_latent: [B, C, T, H, W]
        # mask_latent: [B, C, T, H, W]
        # 官方: concat 应该是沿 channel (dim=1)
        y_concat = torch.cat([ref_vid_latent, mask_latent], dim=1)  # -> [B, 2C, T, H, W]
        logging.info(f"[DEBUG] y_concat: {y_concat.shape}, range:[{y_concat.min():.3f}, {y_concat.max():.3f}]")
        logging.info(f"[DEBUG] pose_embedding: {pose_embedding.shape}")
        
        # 使用 ComfyUI 的条件注入机制
        # 使用 c_concat 代替 y 以避免与标准字段冲突
        positive = node_helpers.conditioning_set_values(
            positive, 
            {
                "c_concat": y_concat,                # 视频+遮罩 (拼接) - 会自动与 x 拼接
                "img_ref": ref_img_latent,           # 参考人脸 (单帧)
                "pose_embedding": pose_embedding,    # 姿态 (原始帧)
            }
        )
        
        negative = node_helpers.conditioning_set_values(
            negative, 
            {
                "c_concat": y_concat,                # 保持相同
                "img_ref": ref_img_latent,           # 保持相同（CFG 时会在模型中零化）
                "pose_embedding": pose_embedding,    # 保持相同
            }
        )
        
        # Debug: verify conditioning structure
        if len(positive) > 0 and len(positive[0]) > 1:
            logging.info(f"[CondPrep DEBUG] positive[0][1] keys: {list(positive[0][1].keys())}")
            logging.info(f"[CondPrep DEBUG] Has img_ref: {'img_ref' in positive[0][1]}")
            logging.info(f"[CondPrep DEBUG] Has pose_embedding: {'pose_embedding' in positive[0][1]}")
            logging.info(f"[CondPrep DEBUG] Has y: {'y' in positive[0][1]}")
        
        
        # 6. Create empty latent for sampling
        spacial_scale = 8  # Wan VAE spatial compression
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
            device=device
        )
        
        out_latent = {"samples": latent}
        
        return io.NodeOutput(positive, negative, out_latent)
