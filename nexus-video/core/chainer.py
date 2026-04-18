import os
import torch
import numpy as np
from PIL import Image
from .generator import VideoGenerator
from .scene_planner import ScenePlanner
import tqdm

class VideoChainer:
    def __init__(self, generator: VideoGenerator, planner: ScenePlanner, temp_dir="nexus-video/temp/clips"):
        self.generator = generator
        self.planner = planner
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def generate_long_video(self, user_prompt, duration_seconds, width=1280, height=720, steps=40, seed=-1, progress=None):
        shot_list = self.planner.plan_scene(user_prompt, duration_seconds)
        all_clips = []

        anchor_frame = None
        prev_frame = None

        if progress:
            progress(0, desc="Starting generation...")

        for i, shot in enumerate(shot_list):
            if progress:
                progress((i / len(shot_list)), desc=f"Generating clip {i+1}/{len(shot_list)}: {shot['prompt'][:50]}...")

            print(f"Generating Shot {i+1}/{len(shot_list)}: {shot['prompt']}")

            # Wan2.2 generates ~5s at 81 frames (24fps * 5s + 1)
            num_frames = 81

            if i == 0:
                # First clip is T2V
                frames = self.generator.generate_t2v(
                    prompt=shot['prompt'],
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_steps=steps,
                    seed=seed
                )
                anchor_frame = frames[0] # Use first frame as anchor
            else:
                # Continuation clips are I2V
                input_frame = prev_frame

                # Every 5th clip, blend with anchor to maintain consistency
                if i % 5 == 0 and anchor_frame is not None:
                    input_frame = self.blend_frames(anchor_frame, prev_frame, anchor_weight=0.3)

                frames = self.generator.generate_i2v(
                    image=input_frame,
                    prompt=shot['prompt'],
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_steps=steps,
                    seed=seed
                )

            # Save the last frame for the next clip
            prev_frame = frames[-1]

            # Convert to numpy and store
            clip_np = self.generator.frames_to_numpy(frames)
            all_clips.append(clip_np)

            # Optional: save intermediate clip
            # self.save_clip(clip_np, i)

        if progress:
            progress(1.0, desc="All clips generated.")

        return all_clips

    def blend_frames(self, frame_a, frame_b, anchor_weight=0.3):
        # Convert PIL to numpy
        a = np.array(frame_a).astype(np.float32)
        b = np.array(frame_b).astype(np.float32)

        # Linear blend
        blended = a * anchor_weight + b * (1.0 - anchor_weight)
        return Image.fromarray(blended.astype(np.uint8))

    def save_clip(self, frames_np, index):
        # Implementation to save clip to disk if needed
        pass
