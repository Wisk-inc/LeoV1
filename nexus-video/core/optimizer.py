import torch
import os

class Optimizer:
    def __init__(self, pipe=None):
        self.pipe = pipe

    def apply_optimizations(self, pipe):
        self.pipe = pipe

        # xFormers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled.")
        except Exception as e:
            print(f"xFormers not available: {e}")

        # VAE Tiling
        try:
            self.pipe.enable_vae_tiling()
            print("VAE tiling enabled.")
        except Exception as e:
            print(f"VAE tiling not available: {e}")

        # CPU Offload (optional, based on VRAM)
        # self.pipe.enable_model_cpu_offload()

        return self.pipe

    def load_lightx2v_lora(self, pipe, lora_path="Vchitect/LightX2V"):
        """
        Load LightX2V LoRA for 4-step inference.
        """
        print(f"Loading LightX2V LoRA from {lora_path}...")
        try:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=1.0)
            print("LightX2V LoRA loaded and fused.")
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
        return pipe

    @staticmethod
    def get_optimal_settings(vram_gb):
        if vram_gb < 12:
            return {"model": "ti2v-5B", "precision": "fp8", "steps": 4}
        elif vram_gb < 16:
            return {"model": "t2v-A14B", "precision": "fp16", "steps": 20}
        else:
            return {"model": "t2v-A14B", "precision": "bf16", "steps": 40}
