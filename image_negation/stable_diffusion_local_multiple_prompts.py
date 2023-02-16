import torch
from diffusers import DiffusionPipeline

def run_model(prompts:list, img_folder:str):
    #code from https://huggingface.co/CompVis/stable-diffusion-v1-4

    model_id = "CompVis/stable-diffusion-v1-4"
    model_path = "/share/kuran/models/huggingface/stable-diffusion-v1-4"
    device = "cuda"

    pipe = DiffusionPipeline.from_pretrained(model_path)
    #pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to(device)

    for prompt_line in prompts:
        prompt = prompt_line.strip() #get rid of any newlines
        image = pipe(prompt).images[0]   
        prompt_as_file_path = prompt.replace(' ', '_')
        image.save(img_folder + "/" + prompt_as_file_path + ".png")

singular_sentences_file = open('non_negated_sentences_singular.txt')
plural_sentences_file = open('non_negated_sentences_plural.txt')
negated_singular_sentences_file = open('negated_sentences_singular.txt')
negated_plural_sentences_file = open('negated_sentences_plural.txt')

singular_sentences = singular_sentences_file.readlines()
negated_singular_sentences = negated_singular_sentences_file.readlines()
plural_sentences = plural_sentences_file.readlines()
negated_plural_sentences = negated_plural_sentences_file.readlines()

singular_images_folder = 'singular_images_stable_diffusion'
negated_singular_images_folder = 'negated_singular_images_stable_diffusion'

def main():
    run_model(negated_singular_sentences, negated_singular_images_folder)

main()




