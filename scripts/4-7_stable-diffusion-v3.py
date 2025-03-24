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
# # Stable Diffusion v3 (SDv3) の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-7_stable-diffusion-v3.ipynb)

# %% [markdown]
# 参考: https://hf.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen[sd3,quantization]

# %%
import logging
import warnings

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
dtype = torch.float16
seed = 42

warnings.simplefilter("ignore", FutureWarning)

# error ログのみを表示する
logger_name = "diffusers.pipelines.pipeline_utils"
logging.getLogger(logger_name).setLevel(logging.ERROR)


# %% [markdown]
# ## Hugging Face Hub へのログイン

# %%
from huggingface_hub import notebook_login

notebook_login()

# %% [markdown]
# ## Stable Diffusion v3 の読み込み

# %%
from diffusers import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, torch_dtype=dtype
)
pipe = pipe.to(device)

# %% [markdown]
# ## ハイパーパラメータの指定

# %%
prompt = (
    "a photo of a cat holding a sign that says hello world"
)
negative_prompt = ""

num_inference_steps = 28
width, height = 1024, 1024
guidance_scale = 7.0

# %% [markdown]
# ## パイプラインによる推論

# %%
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    generator=torch.manual_seed(seed),
)
image = output.images[0]
image

# %%
import gc

pipe = pipe.to("cpu")
del pipe
gc.collect()
torch.cuda.empty_cache()


# %% [markdown]
# ## SDv3 における少ないVRAMによる推論

# %% [markdown]
# ### CPU Offload による推論の実行

# %%
pipe_offload = StableDiffusion3Pipeline.from_pretrained(
    model_id, torch_dtype=dtype
)
pipe_offload.enable_model_cpu_offload()

# %%
output = pipe_offload(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    generator=torch.manual_seed(seed),
)
image_offload = output.images[0]
image_offload

# %%
pipe_offload = pipe_offload.to("cpu")
del pipe_offload
gc.collect()
torch.cuda.empty_cache()


# %% [markdown]
# ### T5 Text Encoder を使用しない推論

# %%
pipe_wo_t5 = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=dtype,
)
pipe_wo_t5 = pipe_wo_t5.to(device)

# %%
output = pipe_wo_t5(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    generator=torch.manual_seed(seed),
)
image_wo_t5 = output.images[0]
image_wo_t5

# %%
pipe_wo_t5 = pipe_wo_t5.to("cpu")
del pipe_wo_t5
gc.collect()
torch.cuda.empty_cache()


# %% [markdown]
# ### T5 Text Encoder の量子化

# %%
from transformers import BitsAndBytesConfig, T5EncoderModel

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

text_encoder_quantized = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
)
pipe_quantized_t5 = (
    StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=text_encoder_quantized,
        device_map="balanced",
        torch_dtype=dtype,
    )
)

# %%
output = pipe_quantized_t5(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    generator=torch.manual_seed(seed),
)
image_quantized_t5 = output.images[0]
image_quantized_t5

# %%
import matplotlib.pyplot as plt

images = {
    "Original": image,
    "CPU Offload": image_offload,
    "Without\nT5 encoder": image_wo_t5,
    "Quantized\n T5 encoder": image_quantized_t5,
}

fig, axes = plt.subplots(
    nrows=1, ncols=len(images), dpi=300
)

for i, (k, image) in enumerate(images.items()):
    axes[i].imshow(image)
    axes[i].set_title(k, fontsize=5)
    axes[i].axis("off")

fig.tight_layout()
