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
# # Safe Latent Diffusion (SLD) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-6-1_safe-latent-diffusion.ipynb)

# %% [markdown]
# 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_safe

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
seed = 42

warnings.simplefilter("ignore", FutureWarning)

# %% [markdown]
# ## Safe Latent Diffusion の実践

# %% [markdown]
# ### Safe Latent Diffusion の読み込み

# %%
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionPipelineSafe,
)

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe_safe = StableDiffusionPipelineSafe.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)
pipe_safe = pipe_safe.to(device)

# %%
print(pipe_safe.safety_concept)

# %%
prompt = "the four horsewomen of the apocalypse, painting by tom of finland, gaston bussiere, craig mullins, j. c. leyendecker"

output_safe = pipe_safe(
    prompt=prompt,
    guidance_scale=6.0,
    generator=torch.manual_seed(seed),
)
image_safe = output_safe.images[0]
image_safe


# %% [markdown]
# ### Stable Diffusion の読み込み

# %%
pipe_unsafe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)
pipe_unsafe = pipe_unsafe.to(device)

# %%
output_unsafe = pipe_unsafe(
    prompt=prompt,
    guidance_scale=6.0,
    generator=torch.manual_seed(seed),
)

image_unsafe = output_unsafe.images[0]
is_nsfw_detected = output_unsafe.nsfw_content_detected[0]

# %%
from PIL import ImageDraw, ImageFont
from PIL.Image import Image as PilImage


def draw_nsfw_warning(image: PilImage) -> PilImage:
    image = image.copy()
    image_w, image_h = image.size
    draw = ImageDraw.Draw(image)
    draw.text(
        xy=(image_w // 2, image_h // 2),
        text="!! NSFW detected !!",
        fill="red",
        anchor="mm",
        font=ImageFont.load_default(size=32),
    )
    return image


# %%
if is_nsfw_detected:
    image_unsafe = draw_nsfw_warning(image_unsafe)

image_unsafe

# %%
import matplotlib.pyplot as plt

images = {
    "Stable Diffusion (Unsafe)": image_unsafe,
    "Safe Latent Diffusion": image_safe,
}

fig, axes = plt.subplots(nrows=1, ncols=len(images))

for ax, (k, v) in zip(axes, images.items()):
    ax.set_title(k)
    ax.imshow(v)
    ax.axis("off")

fig.tight_layout()

# %% [markdown]
# ### Safe Latent Diffusion における安全性の調整

# %%
from typing import List

from diffusers.pipelines.stable_diffusion_safe import (
    SafetyConfig,
)

print(f"WEAK:   {SafetyConfig.WEAK}")
print(f"MEDIUM: {SafetyConfig.MEDIUM}")
print(f"STRONG: {SafetyConfig.STRONG}")
print(f"MAX:    {SafetyConfig.MAX}")

# %%
safety_configs = {
    # "WEAK": SafetyConfig.WEAK, # 不適切な画像が生成しうるので、今回は対象外としました
    "MEDIUM": SafetyConfig.MEDIUM,
    "STRONG": SafetyConfig.STRONG,
    "MAX": SafetyConfig.MAX,
}

generated_images: List[PilImage] = []
for config_type, safety_config in safety_configs.items():
    output = pipe_safe(
        prompt=prompt,
        generator=torch.manual_seed(seed),
        guidance_scale=6.0,
        **safety_config,
    )
    generated_images.extend(output.images)

# %%
fig, axes = plt.subplots(
    nrows=1, ncols=len(generated_images)
)

for ax, config_type, image in zip(
    axes, safety_configs.keys(), generated_images
):
    ax.set_title(f"Safety: {config_type}")
    ax.imshow(image)
    ax.axis("off")

fig.tight_layout()
