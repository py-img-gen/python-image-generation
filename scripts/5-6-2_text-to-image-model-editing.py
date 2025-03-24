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
# # Text-to-Image Model Editing (TIME) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-6-2_text-to-image-model-editing.ipynb)

# %% [markdown]
# - 参考1: https://hf.co/docs/diffusers/main/en/api/pipelines/model_editing
# - 参考2: https://github.com/py-img-gen/diffusers-text-to-model-editing

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
seed = 19950815

warnings.simplefilter("ignore", FutureWarning)

# %% [markdown]
# ### パイプラインの設定

# %%
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

custom_pipeline = (
    "py-img-gen/stable-diffusion-text-to-model-editing"
)

# %% [markdown]
# ### パイプラインの読み込み

# %%
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline=custom_pipeline,
)
pipe = pipe.to(device)

pipe

# %% [markdown]
# ### オリジナルの Stable Diffusion での画像生成

# %%
prompt = "A field of roses"

output = pipe(
    prompt=prompt, generator=torch.manual_seed(seed)
)
image_original = output.images[0]
image_original

# %% [markdown]
# ### TIME を適用した Stable Diffusion での画像生成

# %%
source_prompt = "A pack of roses"
destination_prompt = "A pack of blue roses"

pipe.edit_model(
    source_prompt=source_prompt,
    destination_prompt=destination_prompt,
)

output = pipe(
    prompt=prompt, generator=torch.manual_seed(seed)
)
image_edited = output.images[0]
image_edited

# %% [markdown]
# ### 生成結果の比較

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2, dpi=250)
fig.suptitle(f"Prompt: {prompt}", y=0.92)

axes[0].imshow(image_original)
axes[0].set_title("Original")
axes[0].set_axis_off()

axes[1].imshow(image_edited)
axes[1].set_title(
    f"src: {source_prompt}\ndst: {destination_prompt}"
)
axes[1].set_axis_off()

fig.tight_layout()
