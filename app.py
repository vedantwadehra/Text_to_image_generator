import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
print("Loading models...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def enhance_prompt(prompt, style):
    styles = {
        "Photorealistic": f"{prompt}, photorealistic, 8k, sharp focus, professional photography",
        "Oil Painting":   f"{prompt}, oil painting, impressionist, thick brushstrokes, museum quality",
        "Watercolor":     f"{prompt}, watercolor painting, soft edges, dreamy, artistic",
        "Anime":          f"{prompt}, anime style, Studio Ghibli, vibrant colors, detailed",
        "Sketch":         f"{prompt}, pencil sketch, detailed linework, black and white",
        "Fantasy":        f"{prompt}, fantasy art, magical, ethereal, dramatic lighting, 4k",
    }
    negative = "blurry, bad quality, ugly, distorted, watermark, low resolution"
    return styles.get(style, prompt), negative

def score_image(image, prompt):
    inputs = clip_processor(
        text=[prompt], images=image,
        return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs    = clip_model(**inputs)
        score      = outputs.logits_per_image.item()
        normalized = min(100, max(0, (score / 30) * 100))
    return round(normalized, 1)

def generate(prompt, style, steps, guidance, seed):
    if not prompt.strip():
        return None, "Please enter a prompt!"

    enhanced, negative = enhance_prompt(prompt, style)
    generator = torch.Generator(device).manual_seed(int(seed))

    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        image = pipe(
            enhanced,
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            width=512, height=512,
            generator=generator
        ).images[0]

    score = score_image(image, prompt)
    info  = (
        f"✅ Generated!\n"
        f"📝 Enhanced: {enhanced[:80]}...\n"
        f"🎯 CLIP Score: {score}/100\n"
        f"🔢 Seed: {seed}"
    )
    return image, info

# Gradio UI
with gr.Blocks(title="Text to Image Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎨 Text to Image Generator
    **Built with Stable Diffusion + CLIP | By [Your Name]**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_in   = gr.Textbox(label="Prompt", placeholder="a cat on a windowsill...", lines=3)
            style_in    = gr.Dropdown(
                choices=["Photorealistic","Oil Painting","Watercolor","Anime","Sketch","Fantasy"],
                value="Photorealistic", label="Style"
            )
            steps_in    = gr.Slider(10, 50,  value=30,  step=5,   label="Steps")
            guidance_in = gr.Slider(1,  20,  value=7.5, step=0.5, label="Guidance Scale")
            seed_in     = gr.Number(value=42, label="Seed")
            btn         = gr.Button("🎨 Generate", variant="primary")

        with gr.Column(scale=1):
            img_out  = gr.Image(label="Output", type="pil")
            info_out = gr.Textbox(label="Info", lines=4)

    btn.click(generate,
              inputs=[prompt_in, style_in, steps_in, guidance_in, seed_in],
              outputs=[img_out, info_out])

    gr.Examples(
        examples=[
            ["a house in front of the ocean",         "Photorealistic", 30, 7.5, 42],
            ["a dragon flying over mountains",         "Fantasy",        35, 8.0, 123],
            ["a bowl of ramen noodles",                "Anime",          30, 7.5, 456],
            ["an astronaut on the moon",               "Oil Painting",   35, 8.5, 789],
        ],
        inputs=[prompt_in, style_in, steps_in, guidance_in, seed_in]
    )

app.launch()
