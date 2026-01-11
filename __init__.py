"""
DreamID-V Wrapper for ComfyUI
A clean wrapper that uses DreamIDV.generate() method
"""

from .nodes.model_loader_wrapper import NODE_CLASS_MAPPINGS as LOADER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOADER_NAMES
from .nodes.pose_extractor import NODE_CLASS_MAPPINGS as POSE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as POSE_NAMES
from .nodes.sampler_wrapper import NODE_CLASS_MAPPINGS as SAMPLER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_NAMES

NODE_CLASS_MAPPINGS = {
    **LOADER_MAPPINGS,
    **POSE_MAPPINGS,
    **SAMPLER_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LOADER_NAMES,
    **POSE_NAMES,
    **SAMPLER_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("[DreamID-V Wrapper] Loaded 3 nodes: Model Loader, Pose Extractor, Sampler")
