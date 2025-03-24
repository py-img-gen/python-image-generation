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
# # DDIM (Denoising Diffusion Implicit Models) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-2-2_ddim.ipynb)

# %% [markdown]
# 参考: https://github.com/JeongJiHeon/ScoreDiffusionModel/blob/main/DDIM/DDIM_MNIST.ipynb

# %% [markdown]
# ## 準備
#

# %%
# !pip install -qq py-img-gen

# %%
import pathlib

current_dir = pathlib.Path.cwd()
project_dir = current_dir / "data" / "ddim"
project_dir.mkdir(exist_ok=True, parents=True)
print(f"Created a directory: {project_dir}")

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# %% [markdown]
# ## 設定の定義

# %%
from py_img_gen.trainers import TrainDDPMConfig

train_config = TrainDDPMConfig(
    num_epochs=20, output_dir=project_dir
)
print(train_config)


# %%
from py_img_gen.trainers import EvalConfig

eval_config = EvalConfig()
print(eval_config)


# %%
from py_img_gen.trainers import DDPMModelConfig

model_config = DDPMModelConfig()
print(model_config)


# %% [markdown]
# ## シードの固定

# %%
from transformers import set_seed

set_seed(seed=train_config.seed)

# %% [markdown]
# ## Denoiser の定義

# %%
from dataclasses import asdict

from diffusers import UNet2DModel

unet = UNet2DModel(
    **asdict(model_config),
)
unet = unet.to(device)

# %% [markdown]
# ## Noise Scheduler の定義

# %%
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(
    num_train_timesteps=train_config.num_timesteps,
    beta_start=train_config.beta_1,
    beta_end=train_config.beta_T,
)

# %% [markdown]
# ## Optimizer の定義

# %%
optim = torch.optim.Adam(
    unet.parameters(), lr=train_config.lr
)

# %% [markdown]
# ## データセットの読み込み

# %%
from py_img_gen.trainers import (
    get_simple_resize_transforms,
)

transform = get_simple_resize_transforms(
    sample_size=model_config.sample_size
)
print(transform)

# %%
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.MNIST(
    root=project_dir,
    train=True,
    download=True,
    transform=transform,
)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=train_config.num_workers,
)

# %% [markdown]
# ## DDIM の訓練

# %%
from py_img_gen import trainers

trainers.train(
    train_config=train_config,
    eval_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    optim=optim,
    data_loader=data_loader,
    device=device,
)

# %% [markdown]
# ## DDIM の推論

# %% [markdown]
# ### 推論過程のアニメーションの表示

# %%
from IPython.display import HTML
from py_img_gen import inferencers

ani = inferencers.animation_inference(
    train_config=train_config,
    eval_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
)

HTML(ani.to_jshtml())

# %% [markdown]
# ### $\eta$ の値を変えたときの生成結果の変化

# %%
import dataclasses

ani = inferencers.animation_inference(
    train_config=dataclasses.replace(
        train_config,
        eta_ddim=0.0,  # pure DDIM
    ),
    eval_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
)

HTML(ani.to_jshtml())

# %%
ani = inferencers.animation_inference(
    train_config=dataclasses.replace(
        train_config,
        eta_ddim=1.0,  # pure DDPM
    ),
    eval_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
)
HTML(ani.to_jshtml())

# %%
ani = inferencers.animation_inference(
    train_config=dataclasses.replace(
        train_config,
        eta_ddim=0.5,  # interpolation of DDIM and DDPM
    ),
    eval_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
)

HTML(ani.to_jshtml())

# %% [markdown]
# ### diffusers のパイプラインによる推論

# %%
from diffusers import DDIMPipeline

pipe = DDIMPipeline(unet=unet, scheduler=noise_scheduler)
pipe = pipe.to(device)

# %%
from diffusers.utils import make_image_grid

output = pipe(
    num_inference_steps=train_config.num_timesteps,
    batch_size=eval_config.num_generate_images,
    generator=torch.manual_seed(train_config.seed),
    eta=0.0,  # pure DDIM
)
image = make_image_grid(
    images=output.images,
    rows=eval_config.num_grid_rows,
    cols=eval_config.num_grid_cols,
)
image
