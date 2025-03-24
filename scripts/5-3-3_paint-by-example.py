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
# # Paint-by-Example の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-3-3_paint-by-example.ipynb)

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen

# %%
import logging
import warnings

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
dtype = torch.float16
seed = 19950815

warnings.simplefilter("ignore", FutureWarning)

# error ログを無視する
logger_name = "diffusers.models.modeling_utils"
logging.getLogger(logger_name).setLevel(logging.ERROR)

# %% [markdown]
# ## 動作確認用データ 1 の取得と確認

# %%
from diffusers.utils import load_image, make_image_grid

resize_size = (512, 512)

urls = {
    "init": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png",
    "mask": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png",
    "example": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg",
}
images = {
    k: load_image(url).resize(resize_size)
    for k, url in urls.items()
}
make_image_grid(list(images.values()), rows=1, cols=3)

# %% [markdown]
# ## Paint-by-Example を実装した PaintByExamplePipeline の読み込み

# %%
from diffusers import PaintByExamplePipeline

model_id = "Fantasy-Studio/Paint-by-Example"
pipe = PaintByExamplePipeline.from_pretrained(
    model_id, torch_dtype=dtype
)
pipe = pipe.to(device)

# %% [markdown]
# ## Paint-by-Example による画像編集 1

# %%
edited_image = pipe(
    image=images["init"],
    mask_image=images["mask"],
    example_image=images["example"],
    generator=torch.manual_seed(seed),
).images[0]

images = [images["init"], images["mask"], edited_image]
make_image_grid(images, rows=1, cols=3)

# %% [markdown]
# ## 動作確認用データ 2 の取得と確認

# %%
urls = {
    "init": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_2.png",
    "mask": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_2.png",
    "example": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_2.jpg",
}
images = {
    k: load_image(url).resize(resize_size)
    for k, url in urls.items()
}
make_image_grid(list(images.values()), rows=1, cols=3)

# %% [markdown]
# ## Paint-by-Example による画像編集 2

# %%
edited_image = pipe(
    image=images["init"],
    mask_image=images["mask"],
    example_image=images["example"],
    generator=torch.manual_seed(seed),
).images[0]

images = [images["init"], images["mask"], edited_image]
make_image_grid(images, rows=1, cols=3)

# %% [markdown]
# ## 動作確認用データ 3 の取得と確認

# %%
urls = {
    "init": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_3.png",
    "mask": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_3.png",
    "example": "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_3.jpg",
}
images = {
    k: load_image(url).resize(resize_size)
    for k, url in urls.items()
}
make_image_grid(list(images.values()), rows=1, cols=3)

# %%
edited_image = pipe(
    image=images["init"],
    mask_image=images["mask"],
    example_image=images["example"],
    generator=torch.manual_seed(seed),
).images[0]

images = [images["init"], images["mask"], edited_image]
make_image_grid(images, rows=1, cols=3)
