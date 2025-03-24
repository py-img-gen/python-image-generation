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
# # ControlNet の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-2-2_controlnet.ipynb)

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen[controlnet]

# %%
import warnings

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
dtype = torch.float16
seed = 19950815

warnings.simplefilter("ignore", FutureWarning)

# %% [markdown]
# ## ControlNet を用いた Text-to-Image 生成

# %% [markdown]
# ### オリジナル画像とエッジ画像の準備

# %%
from diffusers.utils import load_image

original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

original_image


# %%
import cv2
import numpy as np
from PIL.Image import Image as PilImage


def get_canny_image(
    image: PilImage,
    low_threshold: float,
    high_threshold: float,
) -> PilImage:
    image_np = np.array(image)

    image_np = cv2.Canny(
        image_np, low_threshold, high_threshold
    )
    # shape: (512, 512) -> (512, 512, 1)
    image_np = image_np[:, :, None]
    # shape: (512, 512, 1) -> (512, 512, 3)
    image_np = np.concatenate(
        [image_np, image_np, image_np], axis=2
    )

    canny_image = Image.fromarray(image_np)
    return canny_image


# %%
from diffusers.utils import make_image_grid
from PIL import Image

canny_image = get_canny_image(
    original_image, low_threshold=100, high_threshold=200
)

make_image_grid(
    [original_image, canny_image], rows=1, cols=2
)

# %% [markdown]
# ### ControlNetModel と StableDiffusionControlNetPipeline の読み込み

# %%
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)

cnet_model_id = "lllyasviel/sd-controlnet-canny"
controlnet = ControlNetModel.from_pretrained(
    cnet_model_id, torch_dtype=dtype, use_safetensors=True
)

pipe_model_id = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    pipe_model_id,
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True,
)

# %% [markdown]
# ### ノイズスケジューラの変更

# %%
from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

# %% [markdown]
# ### CPU offload の有効化

# %%
pipe.enable_model_cpu_offload()

# %% [markdown]
# ### エッジ画像を元にした Text-to-Image 生成結果の表示

# %%
output = pipe(
    prompt="the mona lisa",
    image=canny_image,
    generator=torch.manual_seed(seed),
)
output_image = output.images[0]
images = [original_image, canny_image, output_image]
make_image_grid(images, rows=1, cols=3)

# %% [markdown]
# ## ControlNet を用いた Image-to-Image 生成

# %% [markdown]
# ### 深度マップの取得

# %%
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
)
image


# %%
from typing import Optional

from transformers.pipelines import DepthEstimationPipeline


def get_image_depth_map(
    image: PilImage,
    depth_estimator: Optional[
        DepthEstimationPipeline
    ] = None,
) -> PilImage:
    depth_estimator = depth_estimator or pipeline(
        task="depth-estimation", model="Intel/dpt-large"
    )
    output = depth_estimator(image)
    return output["depth"]


# %%
from transformers import pipeline


def get_tensor_depth_map(
    depth_image: PilImage,
    transform_to_batch: bool = True,
) -> torch.Tensor:
    depth_image_np = np.array(depth_image)
    # shape: (768, 768 -> (768, 768, 1)
    depth_image_np = depth_image_np[:, :, None]
    # shape: (768, 768, 1) -> (768, 768, 3)
    depth_image_np = np.concatenate(
        [depth_image_np, depth_image_np, depth_image_np],
        axis=2,
    )

    depth_image_th = (
        torch.from_numpy(depth_image_np).float() / 255.0
    )
    # shape: (768, 768, 3) -> (3, 768, 768)
    depth_image_th = depth_image_th.permute(2, 0, 1)

    if not transform_to_batch:
        return depth_image_th

    # shape: (3, 768, 768) -> (1, 3, 768, 768)
    depth_image_th = depth_image_th.unsqueeze(dim=0)

    return depth_image_th


# %%
depth_map_pl = get_image_depth_map(image)
depth_map_th = get_tensor_depth_map(depth_map_pl)

# `dtype` の変換と GPU device への移動
depth_map_th = depth_map_th.to(device=device, dtype=dtype)

# %% [markdown]
# ### ControlNetModel と StableDiffusionControlNetImg2ImgPipeline の読み込み

# %%
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
)

cnet_model_id = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(
    cnet_model_id, torch_dtype=dtype, use_safetensors=True
)

pipe_model_id = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    pipe_model_id,
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe.enable_model_cpu_offload()

# %% [markdown]
# ### 深度マップを元にした Image-to-Image 生成結果の表示

# %%
output = pipe(
    prompt="lego batman and robin",
    image=image,
    control_image=depth_map_th,
    generator=torch.manual_seed(seed),
).images[0]

make_image_grid(
    [image, depth_map_pl, output], rows=1, cols=3
)

# %% [markdown]
# ## ControlNet を用いた Inpainting

# %% [markdown]
# ### 初期画像とマスク画像の準備

# %%
init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg"
)
init_image = init_image.resize((512, 512))

mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg"
)
mask_image = mask_image.resize((512, 512))
make_image_grid([init_image, mask_image], rows=1, cols=2)


# %% [markdown]
# ### Inpainting を制御する画像を作成する関数の定義


# %%
def make_inpaint_condition(
    image: PilImage, image_mask: PilImage
) -> torch.Tensor:
    image = image.convert("RGB")
    image_np = np.array(image, dtype=np.float32)
    image_np /= 255.0

    image_mask = image_mask.convert("L")
    image_mask_np = np.array(image_mask, dtype=np.float32)
    image_mask_np /= 255.0

    assert image_np.shape[0:1] == image_mask_np.shape[0:1]
    image_np[
        image_mask_np > 0.5
    ] = -1.0  # マスクされたピクセルとする

    # shape: (512, 512, 3) -> (1, 512, 512, 3)
    image_np = image_np[None, :, :, :]
    # shape: (1, 3, 512, 512)
    image_np = image_np.transpose(0, 3, 1, 2)

    image_th = torch.from_numpy(image_np)
    return image_th


control_image = make_inpaint_condition(
    init_image, mask_image
)

# %% [markdown]
# ### ControlNetModel と StableDiffusionControlNetInpaintPipeline の読み込み

# %%
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
)

cnet_model_id = "lllyasviel/control_v11p_sd15_inpaint"
controlnet = ControlNetModel.from_pretrained(
    cnet_model_id, torch_dtype=dtype, use_safetensors=True
)

pipe_model_id = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    pipe_model_id,
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.enable_model_cpu_offload()

# %% [markdown]
# ### マスク画像を元にした ControlNet による inpainting 結果の表示

# %%
prompt = "corgi face with large ears, detailed, pixar, animated, disney"

output = pipe(
    prompt=prompt,
    num_inference_steps=20,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
    generator=torch.manual_seed(seed),
).images[0]

make_image_grid(
    [init_image, mask_image, output], rows=1, cols=3
)

# %% [markdown]
# ## 複数の条件を考慮可能な MultiControlNet による画像生成

# %% [markdown]
# ### オリジナル画像とエッジ画像の準備

# %%
original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)
image = np.array(original_image)

low_threshold, high_threshold = 100, 200

image = cv2.Canny(image, low_threshold, high_threshold)

# 姿勢情報が重ねられる画像の中央列をゼロにする
zero_start = image.shape[1] // 4
zero_end = zero_start + image.shape[1] // 2
image[:, zero_start:zero_end] = 0

image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
make_image_grid(
    [original_image, canny_image], rows=1, cols=2
)

# %% [markdown]
# ### 姿勢情報の取得

# %%
from controlnet_aux import OpenposeDetector

openpose = OpenposeDetector.from_pretrained(
    "lllyasviel/ControlNet"
)
original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(original_image)
make_image_grid(
    [original_image, openpose_image], rows=1, cols=2
)

# %% [markdown]
# ### 複数の ControlNet を StableDiffusionControlNetPipeline へ渡す

# %%
cn1 = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
)
cn2 = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    pipe_model_id, controlnet=[cn1, cn2], torch_dtype=dtype
)
pipe.scheduer = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe.enable_model_cpu_offload()

# %% [markdown]
# ### 複数の ControlNet による画像生成の結果の表示

# %%
prompt = (
    "a giant standing in a fantasy landscape, best quality"
)
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

control_images = [openpose_image, canny_image]

image = pipe(
    prompt,
    image=control_images,
    num_inference_steps=25,
    generator=torch.manual_seed(seed),
    negative_prompt=negative_prompt,
    controlnet_conditioning_scale=[1.0, 0.8],
).images[0]

make_image_grid(
    [image, openpose_image, canny_image], rows=1, cols=3
)
