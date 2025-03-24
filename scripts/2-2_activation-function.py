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
# # 活性化関数の可視化

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-2_activation-function.ipynb)

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen

# %%
import pathlib

current_dir = pathlib.Path.cwd()
project_dir = current_dir / "data" / "activation-function"
project_dir.mkdir(parents=True, exist_ok=True)

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
plt.rcParams["font.size"] = 20


def visualize(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    bottom: float,
    top: float,
):
    ax.set_xlim(-10, 10)
    ax.set_ylim(bottom, top)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.plot(x, y, linewidth=4)
    return ax


# %%
x = np.arange(-10, 10, 0.01)
x


# %% [markdown]
# ## シグモイド関数


# %%
def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))


y = sigmoid_function(x)

fig, ax = plt.subplots(figsize=(5, 4))
ax = visualize(ax, x, y, bottom=-0.02, top=1.02)
fig.savefig(project_dir / "sigmoid.pdf")


# %% [markdown]
# ## tanh 関数


# %%
def tanh_function(x):
    return (np.exp(x) - np.exp(-x)) / (
        np.exp(x) + np.exp(-x)
    )


y = tanh_function(x)

fig, ax = plt.subplots(figsize=(5, 4))
ax = visualize(ax, x, y, bottom=-1.05, top=1.05)
fig.savefig(project_dir / "tanh.pdf")


# %% [markdown]
# ## ReLU 関数


# %%
def relu_function(x):
    return np.maximum(0, x)


y = relu_function(x)

fig, ax = plt.subplots(figsize=(5, 4))
ax = visualize(ax, x, y, bottom=-0.2, top=10.2)
fig.savefig(project_dir / "relu.pdf")
