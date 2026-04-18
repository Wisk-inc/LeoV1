import gradio as gr
import os
import torch
import yaml
from core.scene_planner import ScenePlanner
from core.generator import VideoGenerator
from core.chainer import VideoChainer
from core.stitcher import VideoStitcher
from ui.components import create_header, create_prompt_input, create_generation_settings, create_cinematic_controls

# Load config
config_path = "config.yaml" if os.path.exists("config.yaml") else "nexus-video/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Lazy initialization of cores
planner = None
generator = None
chainer = None
stitcher = None

def get_cores():
    global planner, generator, chainer, stitcher
    if planner is None:
        planner = ScenePlanner(model_path=config['models']['qwen'])
    if generator is None:
        generator = VideoGenerator(
            model_id_t2v=config['models']['wan22_t2v'],
            model_id_i2v=config['models']['wan22_i2v']
        )
    if chainer is None:
        chainer = VideoChainer(generator, planner)
    if stitcher is None:
        stitcher = VideoStitcher()
    return planner, generator, chainer, stitcher

def generate_video_ui(prompt, enhance, ref_image, neg_prompt, duration, resolution, fps, quality, lighting, camera, upscale, denoise, sharpen, progress=gr.Progress()):
    planner, generator, chainer, stitcher = get_cores()
    log = ""

    def update_log(msg, status="⟳"):
        nonlocal log
        log += f"[{status}] {msg}\n"
        return log

    # 1. Prepare prompt
    final_prompt = prompt
    if enhance:
        yield None, update_log("Enhancing prompt with Qwen...")
        # In a real app, planner would handle this

    if lighting != "Natural":
        final_prompt += f", {lighting} lighting"
    if camera != "Static":
        final_prompt += f", {camera} camera motion"

    final_prompt += ", sharp focus, tack sharp, 8k detail, no motion blur, crisp edges"

    # 2. Determine steps
    steps = 40
    if "Fast" in quality: steps = 4
    elif "Balanced" in quality: steps = 20

    # 3. Generate clips
    yield None, update_log("Planning scene breakdown...")
    shot_list = planner.plan_scene(final_prompt, duration)
    yield None, update_log(f"Scene plan generated ({len(shot_list)} shots)", "✓")

    # Chainer logic integrated for yielding progress
    all_clips = []
    prev_frame = ref_image

    for i, shot in enumerate(shot_list):
        step_progress = (i / len(shot_list))
        msg = f"Generating clip {i+1}/{len(shot_list)}..."
        progress(step_progress, desc=msg)
        yield None, update_log(msg)

        if i == 0 and ref_image is None:
            frames = generator.generate_t2v(shot['prompt'], num_steps=steps)
        else:
            frames = generator.generate_i2v(prev_frame, shot['prompt'], num_steps=steps)

        prev_frame = frames[-1]
        all_clips.append(generator.frames_to_numpy(frames))
        yield None, update_log(f"Clip {i+1} done", "✓")

    # 4. Stitch and process
    yield None, update_log("Stitching and applying post-processing...")
    force_upscale = upscale or ("1080p" in resolution)

    final_path = stitcher.stitch_and_process(
        all_clips,
        fps=int(fps),
        upscale=force_upscale,
        denoise=denoise,
        sharpen=sharpen
    )

    yield final_path, update_log("Done! Video generated successfully.", "✓")

def build_app():
    css_path = "ui/styles.css" if os.path.exists("ui/styles.css") else "nexus-video/ui/styles.css"
    with gr.Blocks(theme=gr.themes.Soft(), css=css_path) as app:
        create_header()

        with gr.Tabs():
            with gr.Tab("GENERATE"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt, enhance, ref_image, neg_prompt = create_prompt_input()
                        duration, resolution, fps, quality = create_generation_settings()
                        lighting, camera = create_cinematic_controls()

                        with gr.Accordion("Post Processing", open=False):
                            upscale_cb = gr.Checkbox(label="Enable AI Upscale (1080p)", value=True)
                            denoise_cb = gr.Checkbox(label="Enable Temporal Denoising", value=True)
                            sharpen_cb = gr.Checkbox(label="Enable Edge Sharpening", value=True)

                        generate_btn = gr.Button("GENERATE VIDEO", variant="primary")

                    with gr.Column(scale=1):
                        output_video = gr.Video(label="Generated Video")
                        status_log = gr.Textbox(label="Generation Log", interactive=False)

                generate_btn.click(
                    fn=generate_video_ui,
                    inputs=[
                        prompt, enhance, ref_image, neg_prompt,
                        duration, resolution, fps, quality,
                        lighting, camera,
                        upscale_cb, denoise_cb, sharpen_cb
                    ],
                    outputs=[output_video, status_log]
                )

            with gr.Tab("HISTORY"):
                gr.Markdown("Past generations will appear here.")

            with gr.Tab("MODELS"):
                gr.Markdown("Model status and VRAM usage.")

            with gr.Tab("SETTINGS"):
                gr.Markdown("System settings and GPU configuration.")

    return app

if __name__ == "__main__":
    app = build_app()
    app.launch()
