import subprocess
import os

class PostProcessor:
    def __init__(self):
        pass

    def apply_filters(self, input_path, output_path, denoise=True, sharpen=True, upscale_to_1080p=True):
        """
        Apply FFmpeg filters for final polish.
        """
        filters = []

        if upscale_to_1080p:
            filters.append("scale=1920:1080:flags=lanczos")

        if denoise:
            # hqdn3d: temporal noise reduction
            filters.append("hqdn3d=3:3:6:6")

        if sharpen:
            # unsharp: edge sharpening
            filters.append("unsharp=5:5:0.8:3:3:0.4")

        filter_str = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"Running FFmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return output_path
