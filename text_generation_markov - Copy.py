from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# User prompt
prompt = input("Enter image prompt: ")

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("generated_image.png")

print("Image generated and saved as generated_image.png")