import torch
from diffusers import DiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
model_path = "/share/kuran/models/huggingface/stable-diffusion-v1-4"
device = "cuda"


pipe = DiffusionPipeline.from_pretrained(model_path)
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to(device)


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
