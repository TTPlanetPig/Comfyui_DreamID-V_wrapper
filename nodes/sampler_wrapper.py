"""
DreamID-V Sampler Wrapper
Uses DreamIDV.generate() method
"""

import torch
import numpy as np
import logging
import os
import uuid
import cv2
from PIL import Image

import folder_paths

# Import from local copy  
from ..dreamidv_wan.configs import SIZE_CONFIGS



# Wan2.1 Latent Mean/Std
WAN21_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 
              0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
WAN21_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 
             3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

class ComfyUIVAEAdapter:
    """
    Adapter to wrap ComfyUI VAE with DreamID-V compatible interface.
    
    DreamID-V expects: vae.encode([video_tensor], device) where video_tensor is [C, T, H, W]
    ComfyUI provides: vae.encode(pixels) where pixels is [B, H, W, C] in [0,1]
    
    The adapter also provides the z_dim attribute and compatible decode() method.
    """
    
    def __init__(self, comfy_vae, device):
        self.comfy_vae = comfy_vae
        self.device = device
        self.z_dim = 16  # Wan 2.1 VAE has 16 latent channels
        
        # Prepare Mean/Std tensors
        self.mean = torch.tensor(WAN21_MEAN, device=device).view(1, 16, 1, 1, 1) # [1, C, 1, 1, 1] for 5D latent
        self.std = torch.tensor(WAN21_STD, device=device).view(1, 16, 1, 1, 1)
        
        # Model reference for compatibility
        self.model = self
    
    def encode(self, videos, device):
        """
        Encode video tensors in DreamID-V format.
        Args: videos list of [C, T, H, W] tensors in [-1, 1]
        Returns: list of encoded latents (Shifted/Model Space)
        """
        results = []
        for video in videos:
            # video is [C, T, H, W] in [-1, 1]
            video = video.to(device)
            
            # [C, T, H, W] -> [T, H, W, C] in [0, 1]
            video_bhwc = video.permute(1, 0, 2, 3).permute(0, 2, 3, 1)
            video_bhwc = (video_bhwc + 1.0) / 2.0
            
            # Encode -> Raw Latent
            latent = self.comfy_vae.encode(video_bhwc)
            
            # Add batch dim if needed [1, C, T, H, W]
            if latent.dim() == 4:
                latent = latent.unsqueeze(0)
            
            # Apply Shift: (Raw - Mean) / Std
            # Ensure mean/std match device
            if self.mean.device != latent.device:
                self.mean = self.mean.to(latent.device)
                self.std = self.std.to(latent.device)
                
            latent = (latent - self.mean) / self.std
            
            # Squeeze batch dim -> [C, T, H, W]
            latent = latent.squeeze(0)
            results.append(latent)
        
        return results
    
    def decode(self, zs):
        """
        Decode latents back to video frames.
        Args: zs list of [C, T, H, W] tensors (Shifted/Model Space)
        Returns: decoded videos [C, T, H, W] in [-1, 1]
        """
        results = []
        for z in zs:
            # z is [C, T, H, W]
            z_batch = z.unsqueeze(0) # [1, C, T, H, W]
            
            # Unshift: Shifted * Std + Mean -> Raw Latent
            if self.mean.device != z_batch.device:
                self.mean = self.mean.to(z_batch.device)
                self.std = self.std.to(z_batch.device)
                
            z_raw = z_batch * self.std + self.mean
            
            # Decode Raw Latent
            decoded = self.comfy_vae.decode(z_raw)
            
            # [T, H, W, C] [0, 1] -> [C, T, H, W] [-1, 1]
            decoded = decoded * 2.0 - 1.0
            decoded = decoded.permute(0, 3, 1, 2).permute(1, 0, 2, 3)
            
            results.append(decoded.clamp(-1, 1))
        
        return results


class DreamIDV_Sampler_Wrapper_TTP:
    """
    Sampler using DreamIDV.generate()
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dreamidv_wrapper": ("DREAMIDV_WRAPPER",),
                "vae": ("VAE",),  # Use ComfyUI's shared VAE loader
                "pose_video": ("IMAGE",),  # From PoseExtractor
                "ref_video": ("IMAGE",),  # Original video frames
                "ref_image": ("IMAGE",),  
                "mask_video": ("IMAGE",),  # From PoseExtractor
                "size": (["832*480", "1280*720", "480*832", "720*1280", "custom"], {"default": "1280*720"}),
                "frame_num": ("INT", {"default": 81, "min": 1, "max": 200, "step": 4}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "offload_t5": ("BOOLEAN", {"default": True, "tooltip": "Offload T5 to CPU after text encoding to save VRAM"}),
                "offload_video_model": ("BOOLEAN", {"default": True, "tooltip": "Offload Video Diffusion Model to CPU after generation"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 720, "min": 64, "max": 2048, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "DreamID-V/Wrapper"
    
    def tensor_to_pil(self, img_tensor):
        """Convert tensor to PIL Image"""
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    
    def save_video(self, frames_tensor, output_path, fps=24):
        """Save frames tensor to video file"""
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        num_frames, height, width, channels = frames_np.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        for i in range(num_frames):
            frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        logging.info(f"[DreamID-V Wrapper] ✓ Saved video: {output_path}")
    
    def sample(self, dreamidv_wrapper, vae, pose_video, ref_video, ref_image, mask_video,
               size, frame_num, steps, cfg_scale, shift, seed, offload_t5, offload_video_model,
               custom_width=1280, custom_height=720):
        """
        Generate video using DreamIDV.generate()
        VAE is passed from ComfyUI's shared VAE loader
        """
        logging.info(f"[DreamID-V Wrapper] Starting generation...")
        logging.info(f"  Frames: {frame_num}, Steps: {steps}, CFG: {cfg_scale}, Shift: {shift}")
        logging.info(f"  Thinking T5: {offload_t5} | Video Model: {offload_video_model}")
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get pipeline and attach external VAE via adapter
        pipeline = dreamidv_wrapper
        # Wrap ComfyUI VAE with adapter for DreamID-V compatible interface
        pipeline.vae = ComfyUIVAEAdapter(vae, device)
        
        # Get size
        if size == 'custom':
            size_tuple = (custom_width, custom_height)
        else:
            size_tuple = SIZE_CONFIGS[size]
        
        logging.info(f"  Size: {size_tuple}")
        
        # Prepare inputs as Tensors (Direct memory passing, no temp files)
        ref_paths = [
            ref_video,
            mask_video,
            ref_image,
            pose_video
        ]
        
        logging.info(f"[DreamID-V Wrapper] Calling pipeline.generate() with in-memory tensors...")
        
        # Generate
        text_prompt = 'change face'
        
        generated = pipeline.generate(
            input_prompt=text_prompt,
            paths=ref_paths,
            size=size_tuple,
            frame_num=frame_num,
            shift=shift,
            sample_solver='unipc',
            sampling_steps=steps,
            guide_scale_img=cfg_scale,
            seed=seed,
            offload_t5=offload_t5,
            offload_video_model=offload_video_model,
            return_latent=True  # Return raw latent for external VAE decode
        )
        
        logging.info(f"[DreamID-V Wrapper] Generated latent: {generated.shape}")
        
        # Convert to ComfyUI LATENT format [B, C, T, H, W]
        # generated is [C, T, H, W], need to add batch dimension
        latent = generated.unsqueeze(0)  # [1, C, T, H, W]
        
        # Unshift output latent for ComfyUI VAE (which expects Raw Latent, not Shifted)
        # Shifted -> * Std + Mean -> Raw
        device = latent.device
        mean = torch.tensor(WAN21_MEAN, device=device).view(1, 16, 1, 1, 1)
        std = torch.tensor(WAN21_STD, device=device).view(1, 16, 1, 1, 1)
        
        latent = latent * std + mean
        
        logging.info(f"[DreamID-V Wrapper] ✓ Complete: LATENT {latent.shape} (Unshifted for ComfyUI VAE)")
        
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "DreamIDV_Sampler_Wrapper_TTP": DreamIDV_Sampler_Wrapper_TTP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamIDV_Sampler_Wrapper_TTP": "DreamID-V Sampler (Wrapper)"
}
