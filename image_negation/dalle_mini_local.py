#Code from https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb#scrollTo=uzjAM2GBYpZX
import jax
import jax.numpy as jnp

from flax.jax_utils import replicate

from functools import partial

# Load models & tokenizer
#import dalle_mini #get the model from huggingface
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

import random
from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

def run_model(prompts:list, output_folder:str):
    #using dalle mini
    DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
    DALLE_COMMIT_ID = None

    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


    # check how many devices are available
    jax.local_device_count()


    # Load dalle-mini
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )

    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )

    #helps the model run faster

    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )


    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    #Preparing text for image generation
    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)

    #Generating images with input text:
    # number of predictions per prompt
    n_predictions = 1 #I changed this to 1

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    #print(f"Prompts: {prompts}\n")
    # generate images
    images = []
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))

        img_index = 0
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
            # display(img)
            # print()

        for img, prompt in zip(images, prompts):
            prompt_as_path = prompt.replace(' ', '_') #create a file path for images
            img.save(output_folder + '/' + prompt_as_path + '.png')

singular_sentences_file = open('non_negated_sentences_singular.txt')
plural_sentences_file = open('non_negated_sentences_plural.txt')
negated_singular_sentences_file = open('negated_sentences_singular.txt')
negated_plural_sentences_file = open('negated_sentences_plural.txt')

singular_sentences = singular_sentences_file.readlines()
negated_singular_sentences = negated_singular_sentences_file.readlines()
plural_sentences = plural_sentences_file.readlines()
negated_plural_sentences = negated_plural_sentences_file.readlines()

singular_images_folder = 'singular_images_dalle_mini'
negated_singular_images_folder = 'negated_singular_images_dalle_mini'

print(len(singular_sentences))

run_model(negated_singular_sentences[200:], negated_singular_images_folder)
