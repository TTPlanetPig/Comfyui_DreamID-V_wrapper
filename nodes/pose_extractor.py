"""
DreamID-V Pose Extractor Node
Extracts pose keypoints and face mask from reference video using MediaPipe
"""
import torch
import logging
import os

# Suppress Mediapipe logs (must be done before importing mediapipe)
os.environ["GLOG_minloglevel"] = "2"

from ..utils.mediapipe_utils import generate_pose_video, generate_mask_video


class DreamIDV_PoseExtractor_TTP:
    """Extract pose and mask from reference video for DreamID-V"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_video": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "max_frames": ("INT", {"default": 81, "min": 5, "max": 999, "step": 4}),
                "static_mode": ("BOOLEAN", {"default": True}),
                "min_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("pose_video", "mask_video", "actual_frames")
    FUNCTION = "extract_pose"
    CATEGORY = "DreamID-V_TTP"
    
    def extract_pose(self, reference_video, reference_image, max_frames, static_mode, 
                     min_detection_confidence, min_tracking_confidence):
        """
        Extract pose and mask from video
        
        Args:
            reference_video: Input video tensor [F, H, W, C], range [0, 1], RGB
            reference_image: Reference face image [1, H, W, C], range [0, 1], RGB
            max_frames: Maximum number of frames to process
            static_mode: True for image mode, False for video tracking mode
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            
        Returns:
            pose_video: Pose visualization [F', H, W, C]
            mask_video: Mask visualization [F', H, W, C]
            actual_frames: Number of frames processed
        """
        try:
            # Convert from ComfyUI tensor format [F, H, W, C] to numpy
            video_np = reference_video.cpu().numpy()
            ref_image_np = reference_image[0].cpu().numpy()  # Take first frame
            
            # Ensure max_frames is 4n+1 format
            adjusted_frames = ((max_frames - 1) // 4) * 4 + 1
            if adjusted_frames != max_frames:
                logging.info(f"[DreamIDV_PoseExtractor] Adjusted max_frames from {max_frames} to {adjusted_frames} (4n+1 format)")
            max_frames = adjusted_frames
            
            # Generate pose and mask videos with all parameters
            pose_np = generate_pose_video(
                video_np, ref_image_np, max_frames, 
                static_mode=static_mode,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            mask_np = generate_mask_video(
                video_np, ref_image_np, max_frames,
                static_mode=static_mode,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            # Convert back to tensors
            pose_tensor = torch.from_numpy(pose_np).float()
            mask_tensor = torch.from_numpy(mask_np).float()
            
            actual_frames = pose_tensor.shape[0]
            
            # Simple completion message
            logging.info(f"[DreamIDV_PoseExtractor] ✓ Pose detection done / Mask gen done ({actual_frames} frames)")
            
            return (pose_tensor, mask_tensor, actual_frames)
            
        except Exception as e:
            logging.error(f"[DreamIDV_PoseExtractor] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node registration
NODE_CLASS_MAPPINGS = {
    "DreamIDV_PoseExtractor_TTP": DreamIDV_PoseExtractor_TTP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamIDV_PoseExtractor_TTP": "DreamID-V Pose Extractor"
}
