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
# # NCSN (Noise Conditional Score Network) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-3_ncsn.ipynb)

# %% [markdown]
# 参考: https://github.com/JeongJiHeon/ScoreDiffusionModel/blob/main/NCSN/NCSN_MNIST.ipynb

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen[ncsn]

# %%
import pathlib

current_dir = pathlib.Path.cwd()
project_dir = current_dir / "data" / "ncsn"
project_dir.mkdir(exist_ok=True, parents=True)
print(f"Created a directory: {project_dir}")

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# %% [markdown]
# ## 設定の定義

# %%
from py_img_gen.trainers import TrainNCSNConfig

train_config = TrainNCSNConfig(
    num_epochs=150,
    num_timesteps=10,
    num_annealed_timesteps=100,
    output_dir=project_dir,
)
print(train_config)


# %%
from py_img_gen.trainers import EvalConfig

eval_config = EvalConfig(eval_epoch=10)
print(eval_config)


# %%
from py_img_gen.trainers import NCSNModelConfig

model_config = NCSNModelConfig()
print(model_config)

# %% [markdown]
# ## シードの固定

# %%
from transformers import set_seed

set_seed(seed=train_config.seed)

# %% [markdown]
# ## Score Network の定義

# %%
from dataclasses import asdict

from ncsn.unet import UNet2DModelForNCSN

unet = UNet2DModelForNCSN(
    num_train_timesteps=train_config.num_timesteps,
    **asdict(model_config),
)
unet = unet.to(device)

# %% [markdown]
# ## Annealed Langevin Dynamics Scheduler の定義

# %%
from ncsn.scheduler import (
    AnnealedLangevinDynamicsScheduler,
)

noise_scheduler = AnnealedLangevinDynamicsScheduler(
    num_train_timesteps=train_config.num_timesteps,
    num_annealed_steps=train_config.num_annealed_timesteps,
    sigma_min=model_config.sigma_min,
    sigma_max=model_config.sigma_max,
    sampling_eps=train_config.sampling_eps,
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
# ## NCSN の訓練

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
# ## NCSN の推論

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
# ### diffusers のパイプラインによる推論

# %%
from ncsn.pipeline_ncsn import NCSNPipeline

pipe = NCSNPipeline(unet=unet, scheduler=noise_scheduler)
pipe = pipe.to(device)

# %%
from diffusers.utils import make_image_grid

output = pipe(
    num_inference_steps=train_config.num_timesteps,
    batch_size=eval_config.num_generate_images,
    generator=torch.manual_seed(train_config.seed),
)
image = make_image_grid(
    images=output.images,
    rows=eval_config.num_grid_rows,
    cols=eval_config.num_grid_cols,
)
image
