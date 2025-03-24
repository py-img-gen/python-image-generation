# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stable Diffusion v2 (SDv2) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-5_stable-diffusion-v2.ipynb)

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen

# %%
import warnings

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
dtype = torch.float16
variant = "fp16"
seed = 11

warnings.simplefilter("ignore", FutureWarning)

# %% [markdown]
# ## Text-to-image
# - 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2#text-to-image

# %% [markdown]
# ### パイプラインの読み込み

# %%
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)

model_id = "stabilityai/stable-diffusion-2"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe = pipe.to(device)

# %% [markdown]
# ### テキストから画像を生成する

# %%
prompt = "High quality photo of an astronaut riding a horse in space"

output = pipe(
    prompt,
    num_inference_steps=25,
    generator=torch.manual_seed(seed),
)
image = output.images[0]

image

# %%
print(image.size)

# %% [markdown]
# ## Inpainting
# - 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2#inpainting

# %% [markdown]
# ### パイプラインの読み込み

# %%
from diffusers import StableDiffusionInpaintPipeline

model_id = "stabilityai/stable-diffusion-2-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe = pipe.to(device)

# %% [markdown]
# ### プロンプトを利用したマスク領域の再構成

# %%
from diffusers.utils import load_image, make_image_grid

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=25,
    generator=torch.manual_seed(seed),
).images[0]

make_image_grid(
    [init_image, mask_image, image], rows=1, cols=3
)

# %% [markdown]
# ## Super-resolution
#
# - 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2#super-resolution

# %% [markdown]
# ### パイプラインの読み込み

# %%
from diffusers import StableDiffusionUpscalePipeline

model_id = "stabilityai/stable-diffusion-x4-upscaler"

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)
pipe = pipe.to(device)

# %% [markdown]
# ### 対象画像の超解像

# %%
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
low_res_img = load_image(url)
low_res_img = low_res_img.resize((128, 128))

prompt = "a white cat"

upscaled_image = pipe(
    prompt=prompt,
    image=low_res_img,
    generator=torch.manual_seed(seed),
).images[0]

images = [
    low_res_img.resize((512, 512)),
    upscaled_image.resize((512, 512)),
]
make_image_grid(images, rows=1, cols=2)

# %% [markdown]
# ## Depth-to-image
# - 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2#depth-to-image

# %%
from diffusers import StableDiffusionDepth2ImgPipeline

model_id = "stabilityai/stable-diffusion-2-depth"

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)
pipe = pipe.to(device)

# %% [markdown]
# ### 深度を考慮した画像を生成する

# %%
url = (
    "http://images.cocodataset.org/val2017/000000039769.jpg"
)
init_image = load_image(url)

prompt = "two tigers"
negative_prompt = "bad, deformed, ugly, bad anotomy"

image = pipe(
    prompt=prompt,
    image=init_image,
    negative_prompt=negative_prompt,
    strength=0.7,
).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
