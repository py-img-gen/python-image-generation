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
# # Latent Consistency Model (LCM) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-4-2_latent-consistency-model.ipynb)

# %% [markdown]
# 参考: https://hf.co/docs/diffusers/using-diffusers/inference_with_lcm

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
seed = 19950815

warnings.simplefilter("ignore", FutureWarning)

# %% [markdown]
# ## 通常の SDXL による画像生成

# %%
from diffusers import StableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe_slow = StableDiffusionXLPipeline.from_pretrained(
    model_id, torch_dtype=dtype, variant=variant
)
pipe_slow = pipe_slow.to(device)

# %%
prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
image_slow = pipe_slow(
    prompt=prompt,
    num_inference_step=50,
    generator=torch.manual_seed(seed),
).images[0]

image_slow

# %% [markdown]
# ## LCM を導入した SDXL による画像生成

# %%
from diffusers import LCMScheduler, UNet2DConditionModel

# LCM に対応した UNet の読み込み
unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=dtype,
    variant=variant,
)
pipe_fast = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    unet=unet,
    torch_dtype=dtype,
    variant=variant,
)
pipe_fast = pipe_fast.to(device)

# %%
pipe_fast.scheduler = LCMScheduler.from_config(
    pipe_fast.scheduler.config
)

# %%
image_fast = pipe_fast(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=8.0,
    generator=torch.manual_seed(seed),
).images[0]

image_fast

# %% [markdown]
# ## 生成品質の比較

# %% [markdown]
# ### LCM の有無による生成品質の比較

# %%
from diffusers.utils import make_image_grid

make_image_grid([image_slow, image_fast], rows=1, cols=2)

# %% [markdown]
# ### ステップ数を LCM に合わせたときの生成品質の比較

# %%
image_slow = pipe_slow(
    prompt=prompt,
    num_inference_steps=4,
    generator=torch.manual_seed(seed),
).images[0]

# %%
image_fast = pipe_fast(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=8.0,
    generator=torch.manual_seed(seed),
).images[0]

# %%
make_image_grid([image_slow, image_fast], rows=1, cols=2)
