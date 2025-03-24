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
# # GLIGEN の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-5-1_gligen.ipynb)

# %% [markdown]
# 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/gligen

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
seed = 42

image_w, image_h = 512, 512

warnings.simplefilter("ignore", FutureWarning)

# error ログを無視する
logger_name = "diffusers.models.modeling_utils"
logging.getLogger(logger_name).setLevel(logging.ERROR)

# %% [markdown]
# ## サンプル画像の準備

# %%
from diffusers.utils import load_image

input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
)
input_image = input_image.resize((image_w, image_h))
input_image

# %% [markdown]
# ## テキスト指示と inpainting による対象領域に対するオジリナル画像への物体の挿入

# %% [markdown]
# ### GLIGEN inpainting pipeline の読み込み

# %%
from diffusers import StableDiffusionGLIGENPipeline

model_id = "masterful/gligen-1-4-inpainting-text-box"

pipe_inpainting = (
    StableDiffusionGLIGENPipeline.from_pretrained(
        model_id, torch_dtype=dtype
    )
)
pipe_inpainting = pipe_inpainting.to(device)


# %% [markdown]
# ### inpaint 対象の確認


# %%
from typing import List, Tuple

from PIL import ImageDraw
from PIL.Image import Image as PilImage

Bbox = Tuple[int, int, int, int]


def draw_bboxes_phrases(
    bboxes: List[Bbox], phrases: List[str], image: PilImage
) -> PilImage:
    """Draw bounding boxes and phrases on the image.

    Args:
        bboxes (List[Bbox]): List of bounding boxes.
        phrases (List[str]): List of phrases.
        image (PilImage): Input image.

    Returns:
        PilImage: Image with bounding boxes and phrases.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    assert len(bboxes) == len(phrases)
    for bbox, phrase in zip(bboxes, phrases):
        draw.rectangle(bbox, outline="red")
        draw.text((bbox[0], bbox[1]), phrase, fill="red")
    return image


# %%
prompt = "a birthday cake"
phrases = ["a birthday cake"]
bboxes = [[137, 312, 244, 368]]

bbox_image = draw_bboxes_phrases(
    bboxes=bboxes, phrases=phrases, image=input_image
)
bbox_image


# %% [markdown]
# ### inpainting pipeline による生成と結果の確認


# %% [markdown]
# #### bbox の正規化


# %%
import numpy as np


def normalize_bboxes(
    bboxes: List[Bbox], w: int, h: int
) -> List[Bbox]:
    """Normalize bounding boxes to [0, 1].

    Args:
        bboxes (List[Bbox]): List of bounding boxes.
        w (int): Image width.
        h (int): Image height.

    Returns:
        List[Bbox]: Normalized bounding boxes.
    """
    bboxes_np = np.array(bboxes, dtype=float)
    bboxes_np[:, 0::2] /= w
    bboxes_np[:, 1::2] /= h
    return bboxes_np.tolist()


bboxes_normalized = normalize_bboxes(
    bboxes, w=image_w, h=image_h
)

# %% [markdown]
# #### 画像生成と結果の確認

# %%
image = pipe_inpainting(
    prompt=prompt,
    gligen_inpaint_image=input_image,
    gligen_boxes=bboxes_normalized,
    gligen_phrases=phrases,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
    generator=torch.manual_seed(seed),
).images[0]

bbox_image = draw_bboxes_phrases(
    bboxes=bboxes, phrases=phrases, image=image
)
bbox_image

# %%
from diffusers.utils import make_image_grid

make_image_grid(
    [input_image, image, bbox_image], rows=1, cols=3
)

# %% [markdown]
# ## テキストと領域指示による新たな画像の生成

# %% [markdown]
# ### GLIGEN generation pipeline の読み込み

# %%
model_id = "masterful/gligen-1-4-generation-text-box"

pipe_gen = StableDiffusionGLIGENPipeline.from_pretrained(
    model_id, torch_dtype=dtype
)
pipe_gen = pipe_gen.to(device)

# %% [markdown]
# ### generation pipeline による生成と結果の確認

# %%
from PIL import Image

prompt = "a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage"

phrases = [
    "a waterfall",
    "a modern high speed train running through the tunnel",
]
bboxes = [
    [71, 105, 218, 363],
    [254, 222, 436, 372],
]

blank_image = Image.new(
    "RGB", (image_w, image_h), color="white"
)

bbox_image = draw_bboxes_phrases(
    bboxes=bboxes, phrases=phrases, image=blank_image
)
bboxes_normalized = normalize_bboxes(
    bboxes, w=image_w, h=image_h
)

image = pipe_gen(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_boxes=bboxes_normalized,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
    generator=torch.manual_seed(seed),
).images[0]

make_image_grid([bbox_image, image], rows=1, cols=2)
