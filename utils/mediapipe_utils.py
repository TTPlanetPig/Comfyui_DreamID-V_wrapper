"""
MediaPipe face detection and pose extraction utilities for DreamID-V
"""
import cv2
import numpy as np
from typing import Tuple, List
import mediapipe as mp

# Core landmarks for pose visualization (from DreamID-V)
CORE_LANDMARK_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
    300, 293, 334, 296, 336, 156,
]

# Face oval landmarks for mask generation (from DreamID-V)
FACE_OVAL_INDICES = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]


class FaceMeshDetector:
    """MediaPipe FaceMesh detector for extracting facial landmarks"""
    
    def __init__(self, static_mode: bool = True, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe FaceMesh detector
        
        Args:
            static_mode: True for image mode (each frame independent), False for video mode (tracking)
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for face tracking (0.0-1.0)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.static_mode = static_mode
        
        # Initialize FaceMesh with separate confidence thresholds
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def __call__(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect face landmarks in image
        
        Args:
            image: RGB image array [H, W, 3], uint8 range [0, 255]
            
        Returns:
            success: Whether face was detected
            landmarks: Landmark coordinates [478, 2] if success, else None
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Ensure RGB format (MediaPipe expects RGB)
        h, w = image.shape[:2]
        
        # Process image
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return False, None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        landmarks = np.array([
            [lm.x * w, lm.y * h] 
            for lm in face_landmarks.landmark
        ], dtype=np.float32)
        
        return True, landmarks
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class FaceMeshAlign:
    """Align pose landmarks from video to reference image"""
    
    def __call__(self, video_landmarks_list: List[np.ndarray], 
                ref_landmarks: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Align video landmarks to reference image landmarks
        
        Args:
            video_landmarks_list: List of landmark arrays for each frame
            ref_landmarks: Reference image landmarks [478, 2]
            
        Returns:
            success: Whether alignment succeeded
            aligned_landmarks: Aligned landmarks [num_frames, 478, 2]
        """
        if not video_landmarks_list or ref_landmarks is None:
            return False, None
        
        # Simple alignment: use reference landmarks as template
        # In production, you might want to implement:
        # - Similarity transform alignment
        # - Temporal smoothing
        # - Expression transfer
        
        num_frames = len(video_landmarks_list)
        aligned = np.zeros((num_frames, 478, 2), dtype=np.float32)
        
        for i, frame_lm in enumerate(video_landmarks_list):
            if frame_lm is not None:
                # For now, just use the video landmarks directly
                # TODO: Implement proper alignment transformation
                aligned[i] = frame_lm
            else:
                # If detection failed, use previous frame or reference
                if i > 0:
                    aligned[i] = aligned[i-1]
                else:
                    aligned[i] = ref_landmarks
        
        return True, aligned


def generate_pose_video(video_frames: np.ndarray, 
                       ref_image: np.ndarray,
                       max_frames: int = 81,
                       static_mode: bool = True,
                       min_detection_confidence: float = 0.5,
                       min_tracking_confidence: float = 0.5) -> np.ndarray:
    """
    Generate pose visualization video from video frames
    
    Args:
        video_frames: Video frames [F, H, W, C] in RGB, range [0, 1]
        ref_image: Reference face image [H, W, C] in RGB, range [0, 1]
        max_frames: Maximum number of frames to process
        static_mode: True for image mode (recommended), False for video tracking mode
        min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
        min_tracking_confidence: Minimum confidence for face tracking (0.0-1.0)
        
    Returns:
        pose_video: Pose visualization video [F, H, W, C] in RGB, range [0, 1]
    """
    # Convert to uint8 for MediaPipe
    video_uint8 = (video_frames * 255).astype(np.uint8)
    ref_uint8 = (ref_image * 255).astype(np.uint8)
    
    detector = FaceMeshDetector(static_mode=static_mode, 
                               min_detection_confidence=min_detection_confidence,
                               min_tracking_confidence=min_tracking_confidence)
    aligner = FaceMeshAlign()
    
    # Detect reference landmarks
    success, ref_landmarks = detector(ref_uint8)
    if not success:
        raise RuntimeError("No face detected in reference image")
    
    # Detect landmarks in video frames
    num_frames = min(len(video_uint8), max_frames)
    # Ensure frames is 4n+1
    num_frames = ((num_frames - 1) // 4) * 4 + 1
    
    video_landmarks = []
    for i in range(num_frames):
        success, landmarks = detector(video_uint8[i])
        if not success:
            raise RuntimeError(f"No face detected in frame {i}")
        video_landmarks.append(landmarks)
    
    # Align landmarks
    success, aligned_landmarks = aligner(video_landmarks, ref_landmarks)
    if not success:
        raise RuntimeError("Failed to align landmarks")
    
    # Generate pose visualization
    h, w = video_uint8.shape[1:3]
    pose_frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
    
    for frame_idx in range(num_frames):
        frame = pose_frames[frame_idx]
        landmarks = aligned_landmarks[frame_idx]
        
        # Draw core landmarks as white points
        for idx in CORE_LANDMARK_INDICES:
            x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
    
    # Convert back to float [0, 1]
    pose_video = pose_frames.astype(np.float32) / 255.0
    
    return pose_video


def generate_mask_video(video_frames: np.ndarray,
                       ref_image: np.ndarray, 
                       max_frames: int = 81,
                       static_mode: bool = True,
                       min_detection_confidence: float = 0.5,
                       min_tracking_confidence: float = 0.5) -> np.ndarray:
    """
    Generate face mask video from video frames
    
    Args:
        video_frames: Video frames [F, H, W, C] in RGB, range [0, 1]
        ref_image: Reference face image [H, W, C] in RGB, range [0, 1]
        max_frames: Maximum number of frames to process
        static_mode: True for image mode (recommended), False for video tracking mode
        min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
        min_tracking_confidence: Minimum confidence for face tracking (0.0-1.0)
        
    Returns:
        mask_video: Face mask video [F, H, W, C] in RGB, range [0, 1]
    """
    # Convert to uint8 for MediaPipe
    video_uint8 = (video_frames *255).astype(np.uint8)
    ref_uint8 = (ref_image * 255).astype(np.uint8)
    
    detector = FaceMeshDetector(static_mode=static_mode,
                               min_detection_confidence=min_detection_confidence,
                               min_tracking_confidence=min_tracking_confidence)
    aligner = FaceMeshAlign()
    
    # Detect reference landmarks
    success, ref_landmarks = detector(ref_uint8)
    if not success:
        raise RuntimeError("No face detected in reference image")
    
    # Detect landmarks in video frames
    num_frames = min(len(video_uint8), max_frames)
    # Ensure frames is 4n+1
    num_frames = ((num_frames - 1) // 4) * 4 + 1
    
    video_landmarks = []
    for i in range(num_frames):
        success, landmarks = detector(video_uint8[i])
        if not success:
            raise RuntimeError(f"No face detected in frame {i}")
        video_landmarks.append(landmarks)
    
    # Align landmarks
    success, aligned_landmarks = aligner(video_landmarks, ref_landmarks)
    if not success:
        raise RuntimeError("Failed to align landmarks")
    
    # Generate mask visualization
    h, w = video_uint8.shape[1:3]
    mask_frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
    
    for frame_idx in range(num_frames):
        frame = mask_frames[frame_idx]
        landmarks = aligned_landmarks[frame_idx]
        
        # Get face oval points
        face_oval_points = landmarks[FACE_OVAL_INDICES].astype(np.int32)
        
        # Fill convex polygon
        cv2.fillConvexPoly(frame, face_oval_points, color=(255, 255, 255))
    
    # Convert back to float [0, 1]
    mask_video = mask_frames.astype(np.float32) / 255.0
    
    return mask_video
