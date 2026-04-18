import os
import numpy as np
import imageio
from datetime import datetime
from .interpolator import FrameInterpolator
from .postprocessor import PostProcessor
from .upscaler import VideoUpscaler

class VideoStitcher:
    def __init__(self, temp_dir="nexus-video/temp"):
        self.temp_dir = temp_dir
        self.interpolator = FrameInterpolator()
        self.upscaler = VideoUpscaler()
        self.postprocessor = PostProcessor()

    def stitch_and_process(self, clips, output_dir="nexus-video/outputs/final", fps=24, upscale=False, denoise=True, sharpen=True):
        os.makedirs(output_dir, exist_ok=True)

        # 1. Merge clips with interpolation
        full_video = clips[0]
        for i in range(1, len(clips)):
            # Use interpolator to blend seams
            full_video = self.interpolator.interpolate_boundary(full_video, clips[i])

        # 2. Save temporary merged video
        temp_merged_path = os.path.join(self.temp_dir, "merged_temp.mp4")
        imageio.mimwrite(temp_merged_path, full_video, fps=fps, quality=9)

        # 3. Final output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"nexus_{timestamp}.mp4"
        final_path = os.path.join(output_dir, final_filename)

        # 4. Post-process (Upscale, Denoise, Sharpen via FFmpeg)
        self.postprocessor.apply_filters(
            temp_merged_path,
            final_path,
            denoise=denoise,
            sharpen=sharpen,
            upscale_to_1080p=upscale
        )

        return final_path
