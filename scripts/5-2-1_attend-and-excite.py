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
# # Attend-and-Excite の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-2-1_attend-and-excite.ipynb)

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
seed = 42

warnings.simplefilter("ignore")

# %% [markdown]
# ## オリジナルの StableDiffusionPipeline の読み込み

# %%
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe_sd = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=dtype
)
pipe_sd = pipe_sd.to(device)

# %% [markdown]
# ## Attend and Excite を実装した StableDiffusionAttendAndExcitePipeline の読み込み

# %%
from diffusers import StableDiffusionAttendAndExcitePipeline

pipe_ae = (
    StableDiffusionAttendAndExcitePipeline.from_pretrained(
        model_id, torch_dtype=dtype
    )
)
pipe_ae = pipe_ae.to(device)

# %% [markdown]
# ## StableDiffusion での画像生成

# %%
from diffusers.utils import make_image_grid

prompt = "A horse and a dog"

images_sd = pipe_sd(
    prompt,
    num_images_per_prompt=2,
    generator=torch.manual_seed(seed),
).images

# %%
gen_result_sd = make_image_grid(
    images=images_sd, rows=1, cols=2
)
gen_result_sd

# %% [markdown]
# ## Attend and Excite を適用した Stable Diffusion での画像生成

# %%
# `get_indices` 関数を使用して、対象のトークン（horse と dog）のインデックスを調べる
# 2 と 5 がそれぞれ horse と dog であることを確認
print(f"Indicies: {pipe_ae.get_indices(prompt)}")

# %%
# 上記で調べたトークンのインデックスを指定
token_indices = [2, 5]

# Attend-and-Excite パイプラインによって画像を生成
images_ae = pipe_ae(
    prompt,
    num_images_per_prompt=2,
    generator=torch.manual_seed(seed),
    #
    # Additional arguments for Attend-and-Excite
    # 対象のトークンを指定
    #
    token_indices=token_indices,
).images

# %%
gen_result_ae = make_image_grid(
    images=images_ae, rows=1, cols=2
)
gen_result_ae

# %% [markdown]
# ## 生成結果の比較

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(20, 5))
grid = ImageGrid(
    fig,
    rect=111,
    nrows_ncols=(1, 2),
    axes_pad=0.1,
)
fig.suptitle(f"Prompt: {prompt}")

images = [
    gen_result_sd,
    gen_result_ae,
]
titles = [
    r"Stable Diffusion ${\it without}$ Attend-and-Excite",
    r"Stable Diffusion ${\it with}$ Attend-and-Excite",
]
for i, (image, title) in enumerate(zip(images, titles)):
    grid[i].imshow(image)
    grid[i].axis("off")
    grid[i].set_title(title)
