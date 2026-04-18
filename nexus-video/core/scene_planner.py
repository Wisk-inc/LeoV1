import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import os

class ScenePlanner:
    def __init__(self, model_path="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        self.device = device
        print(f"Loading Scene Planner model: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.system_prompt = """
You are a cinematic video scene planner. Your job is to take a short prompt and a target duration, and break it down into a list of 5-second shots for an AI video generator.
Each shot must be detailed and maintain cinematic consistency (lighting, character, style).
Output MUST be a valid JSON list of objects. Each object has:
- shot: shot number (int)
- duration: duration in seconds (usually 5)
- prompt: detailed description for this 5s clip
- continuation: how it follows the previous shot

Example Output:
[
  {"shot": 1, "duration": 5, "prompt": "Establishing shot of a neon city, heavy rain, cinematic lighting, sharp focus", "continuation": "Start of the video"},
  {"shot": 2, "duration": 5, "prompt": "Close up of a woman in a trench coat walking through the neon city, rain on her face, cinematic", "continuation": "follows the city atmosphere"}
]
"""

    def plan_scene(self, user_prompt, duration_seconds):
        if duration_seconds <= 5:
            return [{
                "shot": 1,
                "duration": duration_seconds,
                "prompt": f"{user_prompt}, cinematic lighting, sharp focus, 24fps, photorealistic, high detail",
                "continuation": "Single clip generation"
            }]

        num_shots = (duration_seconds + 4) // 5

        full_prompt = f"Break down this prompt into {num_shots} shots of 5 seconds each (total {duration_seconds}s): \"{user_prompt}\""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": full_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            # Clean up response in case of markdown blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            shots = json.loads(response.strip())
            return shots
        except Exception as e:
            print(f"Error parsing Qwen response: {e}")
            # Fallback
            return [{
                "shot": i+1,
                "duration": 5,
                "prompt": user_prompt,
                "continuation": "fallback"
            } for i in range(num_shots)]

if __name__ == "__main__":
    # Test
    planner = ScenePlanner()
    plan = planner.plan_scene("A dragon flying over a medieval city at sunset", 15)
    print(json.dumps(plan, indent=2))
