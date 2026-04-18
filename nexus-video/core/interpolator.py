import torch
import numpy as np
from PIL import Image
import cv2

class FrameInterpolator:
    def __init__(self, device="cuda"):
        self.device = device
        print("Initializing Frame Interpolator (Advanced Blending)...")

    def interpolate_boundary(self, clip_a, clip_b, num_interp=8):
        """
        Smoothly transition between two clips using optical flow-based blending.
        """
        last_frame = clip_a[-1]
        first_frame = clip_b[0]

        # Calculate optical flow to warp frames for smoother blending
        prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        interp_frames = []
        for i in range(1, num_interp + 1):
            alpha = i / (num_interp + 1)

            # Warp frames based on flow (simplified morphing)
            h, w = flow.shape[:2]
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

            map_prev_x = (map_x - flow[..., 0] * alpha).astype(np.float32)
            map_prev_y = (map_y - flow[..., 1] * alpha).astype(np.float32)

            map_next_x = (map_x + flow[..., 0] * (1 - alpha)).astype(np.float32)
            map_next_y = (map_y + flow[..., 1] * (1 - alpha)).astype(np.float32)

            warped_prev = cv2.remap(last_frame, map_prev_x, map_prev_y, cv2.INTER_LINEAR)
            warped_next = cv2.remap(first_frame, map_next_x, map_next_y, cv2.INTER_LINEAR)

            blended = cv2.addWeighted(warped_prev, 1 - alpha, warped_next, alpha, 0)
            interp_frames.append(blended)

        return np.concatenate([clip_a, np.stack(interp_frames), clip_b], axis=0)

    def boost_fps(self, video_np, factor=2):
        """
        Boost FPS using frame interpolation across the whole video.
        """
        new_video = []
        for i in range(len(video_np) - 1):
            new_video.append(video_np[i])
            # Interpolate 1 frame between each
            alpha = 0.5
            blended = cv2.addWeighted(video_np[i], 1 - alpha, video_np[i+1], alpha, 0)
            new_video.append(blended)
        new_video.append(video_np[-1])
        return np.stack(new_video)
