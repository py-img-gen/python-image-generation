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
# # CLIP の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-2_clip.ipynb)

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen[clip]

# %%
import warnings

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
dtype = torch.float16
seed = 42

warnings.simplefilter("ignore", FutureWarning)

# %% [markdown]
# ## CLIP の動作確認

# %% [markdown]
# ### CLIP モデルの読み込み

# %%
from transformers import CLIPModel, CLIPProcessor

model_id = "openai/clip-vit-large-patch14"

# CLIP モデルの読み込み
model = CLIPModel.from_pretrained(model_id)

# モデルを推論モードにする
# このとき dropout を無効化したり、batch normalization の動作を推論用にする
model.eval()

# %% [markdown]
# ### CLIP 用の前処理 pipeline の読み込み

# %%
processor = CLIPProcessor.from_pretrained(model_id)
processor

# %% [markdown]
# ### CLIP のパラメータ情報の表示

# %%
import numpy as np

num_params = sum(
    [int(np.prod(p.shape)) for p in model.parameters()]
)
input_resolution = model.config.vision_config.image_size
context_length = processor.tokenizer.model_max_length
num_vocab = model.config.text_config.vocab_size

print(f"Model parameters: {num_params:,}")
print(f"Input resolution: {input_resolution}")
print(f"Context length: {context_length}")
print(f"Vocab size: {num_vocab:,}")

# %% [markdown]
# ## CLIPProcessor の動作確認

# %% [markdown]
# ### サンプル画像のダウンロード

# %%
from diffusers.utils import load_image

image = load_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/txt2img/000002025.png"
)
image

# %%
image.size

# %% [markdown]
# ### CLIPProcessor による画像の前処理

# %%
output = processor(images=image, return_tensors="pt")
output

# %%
output["pixel_values"].size()

# %% [markdown]
# ### CLIPProcessor によるテキストの前処理

# %%
output = processor(text="Hello world", return_tensors="pt")
output

# %%
processor.batch_decode(output["input_ids"])

# %% [markdown]
# ## CLIP による画像とテキストの類似度計算
#
# - 参考: https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb

# %% [markdown]
# ### zero-shot 分類用のプロンプトの設定

# %%
#
# 使用する skimage の画像とその説明文
#
descriptions_dict = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer",
}

# %% [markdown]
# ### 画像とテキストのペアの構築

# %%
import os

import skimage
from more_itertools import sort_together

original_imgs, original_txts = [], []

# skimage から .png か .jpg な画像のパスを習得する
filenames = [
    fname
    for fname in os.listdir(skimage.data_dir)
    if fname.endswith(".png") or fname.endswith(".jpg")
]
for fname in filenames:
    name, _ = os.path.splitext(fname)
    if name not in descriptions_dict:
        continue

    # 画像の読み込み
    image_path = os.path.join(skimage.data_dir, fname)
    original_imgs.append(load_image(image_path))
    # テキストの読み込み
    original_txts.append(descriptions_dict[name])

# 画像とテキストの数があっているか確認
assert len(original_txts) == len(original_imgs)

# テキストの文字列をベースに、テキストと画像のリストをソートする
original_txts, original_imgs = sort_together(
    (original_txts, original_imgs)
)

# %% [markdown]
# ### 画像とテキストのペアの可視化

# %%
import matplotlib.pyplot as plt

nrows, ncols = 2, 4
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(16, 5)
)

for i in range(nrows):
    for j in range(ncols):
        axes[i][j].imshow(original_imgs[i * ncols + j])
        axes[i][j].axis("off")
        axes[i][j].set_title(
            original_txts[i * ncols + j], fontsize=10
        )

# %% [markdown]
# ### 画像とテキストのペアの前処理

# %%
inputs = processor(
    text=original_txts,
    images=original_imgs,
    padding="max_length",
    return_tensors="pt",
)
inputs

# %% [markdown]
# ### CLIP による画像とテキストの特徴の取得

# %%
import torch

with torch.no_grad():
    img_features = model.get_image_features(
        pixel_values=inputs["pixel_values"],
    )
    txt_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

# %% [markdown]
# ### 画像とテキストの類似度計算

# %%
img_features = img_features / img_features.norm(
    p=2, dim=-1, keepdim=True
)
txt_features = txt_features / txt_features.norm(
    p=2, dim=-1, keepdim=True
)

similarity = img_features @ txt_features.T

# %% [markdown]
# ### 画像とテキストの類似度の可視化

# %%
assert len(original_imgs) == len(original_txts)
count = len(original_imgs)

fig, ax = plt.subplots(figsize=(20, 14))
ax.imshow(similarity, vmin=0.1, vmax=0.3)

ax.set_yticks(
    range(len(original_txts)),
    labels=original_txts,
    fontsize=18,
)
ax.set_xticks([])

for i, img in enumerate(original_imgs):
    extent = (i - 0.5, i + 0.5, -1.6, -0.6)
    ax.imshow(img, extent=extent, origin="lower")

for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        s = f"{similarity[y, x]:.2f}"
        a = "center"
        ax.text(x, y, s=s, ha=a, va=a, size=12)

for side in ("left", "top", "right", "bottom"):
    plt.gca().spines[side].set_visible(False)

ax.set_xlim((-0.5, count - 0.5))
ax.set_ylim((count - 0.5, -2))

ax.set_title(
    "Cosine similarity between text and image features",
    size=20,
)

# %% [markdown]
# ## CLIP による zero-shot 画像分類

# %% [markdown]
# ### CIFAR100 データセットの読み込み

# %%
from torchvision.datasets import CIFAR100

cifar100 = CIFAR100(
    os.path.expanduser("~/.cache"), download=True
)

# %% [markdown]
# ### プロンプトの準備

# %%
text_template = "This is a photo of a {label}"
text_descriptions = [
    text_template.format(label=label)
    for label in cifar100.classes
]

# %% [markdown]
# ### テキスト特徴の取得

# %%
inputs = processor(
    text=text_descriptions,
    padding="max_length",
    return_tensors="pt",
)

with torch.no_grad():
    txt_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    txt_features = txt_features / txt_features.norm(
        p=2, dim=-1, keepdim=True
    )

# %% [markdown]
# ### 類似度を下にした分類結果の取得

# %%
txt_probs = 100 * img_features @ txt_features.T
txt_probs = txt_probs.softmax(dim=-1)
top_probs, top_labels = txt_probs.topk(5, dim=-1)

# %% [markdown]
# ### zero-shot 分類結果の可視化

# %%
nrows, ncols = 4, 4
fig = plt.figure(figsize=(16, 16))
plt.style.use("ggplot")

y = np.arange(top_probs.shape[-1])

for i, img in enumerate(original_imgs):
    ax1 = fig.add_subplot(nrows, ncols, 2 * i + 1)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(original_txts[i], fontsize=10)

    ax2 = fig.add_subplot(nrows, ncols, 2 * i + 2)
    ax2.barh(y, top_probs[i])

    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    ax2.set_yticks(
        y, [cifar100.classes[idx] for idx in top_labels[i]]
    )
    ax2.set_xlabel("Probability")

fig.subplots_adjust(wspace=0.5)
