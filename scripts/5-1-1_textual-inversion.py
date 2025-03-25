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
# # Textual Inversion の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-1-1_textual-inversion.ipynb)

# %% [markdown]
# 参考: https://huggingface.co/docs/diffusers/en/training/text_inversion

# %% [markdown]
# ## 準備

# %%
# !pip install -qq py-img-gen

# %%
import diffusers

diffusers.utils.logging.set_verbosity_error()


# %%
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
resize_size = 512

# %%
import os

from google.colab import drive

# /content/drive をマウントする
DRIVE_PATH = os.path.join(os.sep, "content", "drive")
print(f"Mount the following directories: {DRIVE_PATH}")

drive.mount(DRIVE_PATH)

#
# 本 notebook 用のデータを格納するディレクトリを作成する
# まずベースとなるディレクトリとして以下のようなディレクリを作成する:
# /content/drive/MyDrive/colab-notebooks/oloso/practice
#
base_dir_path = os.path.join(
    DRIVE_PATH,
    "MyDrive",
    "colab-notebooks",
    "coloso",
    "practice",
)
#
# 次に講義用のディレクトリを作成する。今回は第 20 講なので `lecture-20` と命名する:
# /content/drive/MyDrive/colab-notebooks/coloso/practice/lecture-20
#
lecture_dir_path = os.path.join(base_dir_path, "lecture-20")

#
# 今回使用する学習画像を保存するディレクトリを作成する:
# /content/drive/MyDrive/colab-notebooks/coloso/practice/lecture-20/sample-images
#
sample_image_dir_path = os.path.join(
    lecture_dir_path, "sample-images"
)
print(
    f"The images will be saved in the following path: {sample_image_dir_path}"
)

# 上記のディレクトリが存在しない場合は作成する
if not os.path.exists(sample_image_dir_path):
    os.makedirs(sample_image_dir_path)

# %%
urls = [
    "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/1.jpeg",
    "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/2.jpeg",
    "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/3.jpeg",
    "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/4.jpeg",
    #
    # ここに更に画像を追加することができます
    #
    # "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/5.jpeg",
    # "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/6.jpeg",
    # "https://huggingface.co/datasets/diffusers/cat_toy_example/resolve/main/7.jpeg",
]

# %%
from pathlib import Path

from py_img_gen.utils.download import download_image

for i, url in enumerate(urls):
    image_filepath = os.path.join(
        sample_image_dir_path, f"{i}.jpg"
    )
    print(
        f"The image is saved in the following path: {image_filepath}"
    )
    image = download_image(
        url, save_path=Path(image_filepath)
    )

# %%
from typing import List

from diffusers.utils import make_image_grid
from PIL import Image
from py_img_gen.typehints import PilImage

images: List[PilImage] = []
for file_path in os.listdir(sample_image_dir_path):
    image_filepath = os.path.join(
        sample_image_dir_path, file_path
    )
    image = Image.open(image_filepath)
    image = image.resize((512, 512))
    images.append(image)

make_image_grid(images, rows=1, cols=len(images))

# %%
from typing import Literal, get_args

# モデルに教える特性の選択肢を定義します。ここでは object と style が選択可能
LearnableProperty = Literal["object", "style"]

# モデルに何を教えるかを指定
what_to_teach = "object"

# 新しい概念を表現するために使用するトークンを指定
placeholder_token = ""

# 新しい概念に関連する単語を指定
initializer_token = (
    "toy"  # 今回の場合は cat でもいいかもしれません
)

# `what_to_teach` を正しく設定できているか確認します
if what_to_teach not in get_args(LearnableProperty):
    raise ValueError(
        f"Invalid learnable property: {what_to_teach}. "
        f"You should choose from the following options: {get_args(LearnableProperty)}."
    )

# %%
from transformers import CLIPTokenizer

# tokenizer の読み込み
tokenizer = CLIPTokenizer.from_pretrained(
    model_id, subfolder="tokenizer"
)

# `placeholder_token` と tokenizer に追加
num_added_tokens = tokenizer.add_tokens(placeholder_token)

if num_added_tokens == 0:
    #
    # `placeholder_token` が元々 `tokenizer` に含まれていたらエラーを出す
    # エラーになったら違う `placeholder_token` を指定するようにしてください
    #
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token}. ",
        "Please pass a difference `placeholder_token` that is not already in the tokenizer.",
    )

# %%
token_ids = tokenizer.encode(
    initializer_token, add_special_tokens=False
)
if len(token_ids) > 1:
    raise ValueError(
        "The initializer token must be a single token."
    )

initializer_token_id = token_ids[0]
placeholder_token_id = tokenizer.convert_tokens_to_ids(
    placeholder_token
)

# %%
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel

text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    model_id, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet"
)

# %%
text_encoder.resize_token_embeddings(len(tokenizer))

# %%
token_embeds = (
    text_encoder.get_input_embeddings().weight.data
)
token_embeds[placeholder_token_id] = token_embeds[
    initializer_token_id
]

# %%
import itertools
from typing import Iterable

import torch.nn as nn


def freeze_params(params: Iterable[nn.Parameter]) -> None:
    for param in params:
        param.requires_grad = False


# VAE と U-Net のパラメータを固定
freeze_params(vae.parameters())
freeze_params(unet.parameters())

# Text Encoder において、追加したパラメータ以外を固定
params_to_freeze = itertools.chain(
    text_encoder.text_model.encoder.parameters(),
    text_encoder.text_model.final_layer_norm.parameters(),
    text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)

# %%
from py_img_gen.training.textual_inversion import (
    IMAGENET_STYLE_TEMPLATES_SMALL,
    IMAGENET_TEMPLATES_SMALL,
)

print(f"{IMAGENET_TEMPLATES_SMALL=}")
print(f"{IMAGENET_STYLE_TEMPLATES_SMALL=}")

# %%
from py_img_gen.datasets.textual_inversion import (
    TextualInversionDataset,
)

templates = (
    IMAGENET_TEMPLATES_SMALL
    if what_to_teach == "object"
    else IMAGENET_STYLE_TEMPLATES_SMALL
)

train_dataset = TextualInversionDataset(
    data_root=sample_image_dir_path,
    tokenizer=tokenizer,
    image_size=vae.config.sample_size,
    placeholder_token=placeholder_token,
    repeats=100,
    templates=templates,
    is_center_crop=False,
    split="train",
)

# %%
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)

# %%
from dataclasses import dataclass


@dataclass
class Hyperparameter(object):
    learning_rate: float = 5e-4
    scale_lr: bool = True
    max_train_steps: int = 2000  # デフォルトは 2000 程度
    save_steps: int = 250
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    seed: int = 19950815
    output_dir_path: str = os.path.join(
        lecture_dir_path, "sd-concept-output"
    )


hparams = Hyperparameter()
print(hparams)

print(
    f"The training results are saved in the following directory: {hparams.output_dir_path}"
)
if not os.path.exists(hparams.output_dir_path):
    os.makedirs(hparams.output_dir_path)

# %%
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def save_progress(
    text_encoder: CLIPTextModel,
    placeholder_token_id: int,
    accelerator: Accelerator,
    save_path: str,
) -> None:
    logger.info("Saving embeddings")

    # 新たに追加した概念に対応する埋め込みベクトルのみを保存する
    # `placeholder_token` の ID を指定することで対象のベクトルを取得可能
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    learned_embeds_dict = {
        placeholder_token: learned_embeds.detach().cpu()
    }

    torch.save(learned_embeds_dict, save_path)


# %%
import math

import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm


def training_function(
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
):
    # ハイパーパラメータの値を取得
    train_batch_size = hparams.train_batch_size
    gradient_accumulation_steps = (
        hparams.gradient_accumulation_steps
    )
    learning_rate = hparams.learning_rate
    max_train_steps = hparams.max_train_steps
    # output_dir_path = hparams.output_dir_path
    gradient_checkpointing = hparams.gradient_checkpointing

    # 学習を効率化する Accelerator の設定
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=hparams.mixed_precision,
    )

    # GPU メモリの使用を抑える Gradient Checkpointing の設定
    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # 学習用のデータローダーの設定
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    if hparams.scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # 最適化手法を初期化
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # 追加した placeholder_token の部分のみ学習
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = (
        accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # GPU へ VAE と U-Net を移動させます
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # VAE は学習させないため eval モードに
    vae.eval()
    # U-Net は gradient checkpointing を有効にするため train モードに
    unet.train()

    # 学習用データローダーのサイズが gradient accumulation の数によって変わる可能性があるため
    # ここで再度学習ステップ数を計算し直す
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        max_train_steps / num_update_steps_per_epoch
    )

    # Train!
    total_batch_size = (
        train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(
        f"  Instantaneous batch size per device = {train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {gradient_accumulation_steps}"
    )
    logger.info(
        f"  Total optimization steps = {max_train_steps}"
    )

    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # 画像を潜在データへ変換
                latents = (
                    vae.encode(
                        batch["pixel_values"].to(
                            dtype=weight_dtype
                        )
                    )
                    .latent_dist.sample()
                    .detach()
                )
                latents = latents * 0.18215

                # 潜在データへ追加するノイズを取得
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # 各画像に対してランダムなタイムステップ数を取得
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # 各タイムステップにおけるノイズの大きさに従って
                # 潜在データにノイズを追加（拡散過程）
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # 条件付けのためにプロンプトからテキストベクトルを取得
                encoder_hidden_states = text_encoder(
                    batch["input_ids"]
                )[0]

                # ノイズを予測
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states.to(weight_dtype),
                ).sample

                # 予測タイプに応じた損失を計算
                if (
                    noise_scheduler.config.prediction_type
                    == "epsilon"
                ):
                    target = noise
                elif (
                    noise_scheduler.config.prediction_type
                    == "v_prediction"
                ):
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = (
                    F.mse_loss(
                        noise_pred, target, reduction="none"
                    )
                    .mean([1, 2, 3])
                    .mean()
                )
                accelerator.backward(loss)

                # 概念の埋め込みだけを最適化したいので、
                # 新しく追加された概念の埋め込み以外のすべてのトークンの埋め込みに対する勾配をゼロに
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # 勾配をゼロにする対象のインデックスを取得
                # `placeholder_token_id` 以外のものを選択することで達成
                index_grads_to_zero = (
                    torch.arange(len(tokenizer))
                    != placeholder_token_id
                )
                grads.data[index_grads_to_zero, :] = (
                    grads.data[
                        index_grads_to_zero, :
                    ].fill_(0)
                )

                optimizer.step()
                optimizer.zero_grad()

            # accelerator がバックグラウンドで最適化工程を実行したかを確認
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % hparams.save_steps == 0:
                    save_path = os.path.join(
                        hparams.output_dir_path,
                        f"learned_embeds-step-{global_step}.bin",
                    )
                    save_progress(
                        text_encoder,
                        placeholder_token_id,
                        accelerator,
                        save_path,
                    )

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # 学習したモデルを元に、pipeline を構築して保存
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            text_encoder=accelerator.unwrap_model(
                text_encoder
            ),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(hparams.output_dir_path)
        # 新たに追加した概念に対応するパラメータも保存
        save_path = os.path.join(
            hparams.output_dir_path, "learned_embeds.bin"
        )
        save_progress(
            text_encoder,
            placeholder_token_id,
            accelerator,
            save_path,
        )


# %%
import accelerate

accelerate.notebook_launcher(
    training_function, args=(text_encoder, vae, unet)
)

for param in itertools.chain(
    unet.parameters(), text_encoder.parameters()
):
    if param.grad is not None:
        del param.grad  # Colab では RAM の制約があるため勾配に関する情報を削除
    torch.cuda.empty_cache()

# %%
hparams.output_dir_path = "drive/MyDrive/colab-notebooks/coloso/practice/lecture-20/pre-trained/"

# %%
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(
    hparams.output_dir_path, subfolder="scheduler"
)

pipe = StableDiffusionPipeline.from_pretrained(
    hparams.output_dir_path,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# %%
prompt = "A  backpack"

num_samples = 2
num_rows = 1

# 結果を再現できるように乱数の seed を固定
generator = torch.Generator().manual_seed(42)

all_images = []
for _ in range(num_rows):
    images = pipe(
        prompt,
        num_images_per_prompt=num_samples,
        generator=generator,
        num_inference_steps=25,
    ).images
    all_images.extend(images)

make_image_grid(all_images, num_rows, num_samples)
