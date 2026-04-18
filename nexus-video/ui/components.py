import gradio as gr

def create_header():
    with gr.Row():
        gr.Markdown("""
        # 🌌 NEXUS VIDEO
        ### Cinematic AI. No limits.
        """)

def create_prompt_input():
    with gr.Column():
        prompt = gr.Textbox(
            label="Text Prompt",
            placeholder="Describe your video in detail...",
            lines=4
        )
        enhance_prompt = gr.Checkbox(label="Enhance Prompt with AI (Qwen2.5)", value=True)
        reference_image = gr.Image(label="Reference Image (Optional)", type="pil")
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="blur, distortion, low quality, watermark, grain, noise, artifacts, overexposed, underexposed",
            visible=False
        )
        with gr.Accordion("Advanced Prompting", open=False):
            negative_prompt.visible = True
            gr.Markdown("Add negative constraints here.")

    return prompt, enhance_prompt, reference_image, negative_prompt

def create_generation_settings():
    with gr.Column():
        gr.Markdown("### Generation Settings")
        duration = gr.Slider(minimum=1, maximum=360, value=5, step=1, label="Duration (seconds)")
        resolution = gr.Dropdown(
            choices=["480p", "720p", "1080p (upscaled)"],
            value="720p",
            label="Resolution"
        )
        fps = gr.Dropdown(choices=["24", "48", "60"], value="24", label="Frame Rate")
        quality = gr.Radio(
            choices=["Fast (4-step)", "Balanced (20-step)", "Quality (40-step)"],
            value="Quality (40-step)",
            label="Quality Mode"
        )
    return duration, resolution, fps, quality

def create_cinematic_controls():
    with gr.Accordion("Cinematic Controls", open=False):
        lighting = gr.Dropdown(
            choices=["Natural", "Golden Hour", "Night", "Studio", "Overcast", "Neon", "Cinematic"],
            value="Cinematic",
            label="Lighting Style"
        )
        camera = gr.Dropdown(
            choices=["Static", "Slow Pan", "Dolly In", "Dolly Out", "Orbit", "Handheld", "Aerial"],
            value="Static",
            label="Camera Motion"
        )
    return lighting, camera
