import torch
import numpy as np
from PIL import Image
from diffusers import WanPipeline, WanImageToVideoPipeline
import os

class VideoGenerator:
    def __init__(self, model_id_t2v=None, model_id_i2v=None, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.t2v_pipe = None
        self.i2v_pipe = None
        self.model_id_t2v = model_id_t2v
        self.model_id_i2v = model_id_i2v

    def load_t2v(self):
        if self.t2v_pipe is None:
            print(f"Loading T2V model: {self.model_id_t2v}...")
            self.t2v_pipe = WanPipeline.from_pretrained(
                self.model_id_t2v,
                torch_dtype=self.dtype
            )
            self.t2v_pipe.to(self.device)
            # self.t2v_pipe.enable_model_cpu_offload() # Use if VRAM is tight
            self.t2v_pipe.enable_vae_tiling()

    def load_i2v(self):
        if self.i2v_pipe is None:
            print(f"Loading I2V model: {self.model_id_i2v}...")
            self.i2v_pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id_i2v,
                torch_dtype=self.dtype
            )
            self.i2v_pipe.to(self.device)
            # self.i2v_pipe.enable_model_cpu_offload() # Use if VRAM is tight
            self.i2v_pipe.enable_vae_tiling()

    def generate_t2v(self, prompt, negative_prompt=None, num_frames=81, width=1280, height=720, num_steps=40, guidance_scale=5.0, seed=-1):
        self.load_t2v()
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None

        output = self.t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # frames is a list of PIL images or a numpy array/tensor depending on pipeline
        # Usually for Wan in diffusers, it's a list of PIL images
        return output.frames[0] # List of frames

    def generate_i2v(self, image, prompt, negative_prompt=None, num_frames=81, width=1280, height=720, num_steps=40, guidance_scale=5.0, seed=-1):
        self.load_i2v()
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None

        # Ensure image is PIL and resized if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        output = self.i2v_pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        return output.frames[0]

    @staticmethod
    def frames_to_numpy(frames):
        if isinstance(frames[0], Image.Image):
            return np.stack([np.array(f) for f in frames])
        return frames
