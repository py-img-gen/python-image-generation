# Pythonで学ぶ画像生成（機械学習実践シリーズ）

[![CI](https://github.com/py-img-gen/python-image-generation/actions/workflows/ci.yaml/badge.svg)](https://github.com/py-img-gen/python-image-generation/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Appach--2.0-blue)](https://github.com/py-img-gen/python-image-generation/blob/main/LICENSE)
![Python](https://img.shields.io/badge/🐍%20Python-3.10+-orange)
[![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.32.0-orange)](https://github.com/huggingface/diffusers)

<img align="right" width="30%" src="https://github.com/user-attachments/assets/41bf761b-b55c-49d9-a273-df34c68f4a4b">

本レポジトリではインプレス社より出版されている　[北田 俊輔](https://shunk031.me/) 著 の機械学習シリーズ「[Pythonで学ぶ画像生成](https://book.impress.co.jp/books/1123101104)」で扱うソースコードを管理しています。
ソースコードは Jupyter Notebook 形式でまとめられており、Google Colab 等で実行することを想定しています。

ソースコードの解説は主に書籍内に記載されており、本レポジトリのソースコードは補助教材となっています。

本書籍は「**画像生成の基礎から実践までを一冊に凝縮**」というテーマで各章が構成されています。
まず​「画像生成とは​何か」と​いう​基本を​解説し、​次に​画像生成を​支える​深層学習の​基礎を​押さえます。​その上で、​現在の​最先端技術である​拡散モデルと、​その​効率化・応用例と​して​Stable Diffusionなどを​詳しく​取り上げています。​最後には、​拡散モデルが​もたらす革新的な​可能性と​同時に、​技術の​制限や​倫理的な​課題にも​言及し、​将来の​さらなる​発展・応用に​向けた​展望を​示しています。

-  6章 + 各章末に実装に役立つコラム付き
-  Python・PyTorchで学ぶ画像生成の実装
-  Diffusersによる最先端技術の実践
-  画像生成を中心とした様々なタスクの解説を多数収録

## 📄 書籍の内容と補助教材

Jupyter Notebook の補助教材があるセクションには `Open in Colab` のバッジを付与しています。バッジをクリックすると該当するノートブックを Colab で開けます。

### 第 1 章: 画像生成とは？

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 画像生成の概要 | --- |  --- |
| 2. テキストからの画像生成 | ![Static Badge](https://img.shields.io/badge/GitHub-Text--to--Image-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/1-2_text-to-image-generation.ipynb) |
| 3. 画像生成技術の進歩による弊害 | --- | --- |

### 第 2 章: 深層学習の基礎知識

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 深層学習の概要 | --- | --- |
| 2. 深層学習の訓練と評価 | ![Static Badge](https://img.shields.io/badge/GitHub-PyTorch-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-2_pytorch.ipynb) |
| 3. 注意機構と Transformer モデル | ![Static Badge](https://img.shields.io/badge/GitHub-Transformer-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |

### 第 3 章: 拡散モデルの導入

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 生成モデル | --- | --- |
| 2. DDPM（ノイズ除去拡散確率モデル）| ![Static Badge](https://img.shields.io/badge/GitHub-DDPM-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-2-1_ddpm.ipynb) |
|                               | ![Static Badge](https://img.shields.io/badge/GitHub-DDIM-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-2-2_ddim.ipynb) |
| 3. スコアベース生成モデル | ![Static Badge](https://img.shields.io/badge/GitHub-NCSN-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-3_ncsn.ipynb) |
| 4. 拡散モデルの生成品質の向上 | ![Static Badge](https://img.shields.io/badge/GitHub-CFG-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-4_classifier-free-guidance.ipynb) |

### 第 4 章: 潜在拡散モデルと Stable Diffusion

| Section | GitHub | Colab |
|---------|---------|------|
| 1. LDM（潜在拡散確率モデル） | --- | --- |
| 2. CLIP | ![Static Badge](https://img.shields.io/badge/GitHub-CLIP-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-2_clip.ipynb) |
| 3. Stable Diffusion を構成する要素 | ![Static Badge](https://img.shields.io/badge/GitHub-SD--Components-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 4. Stable Diffusion v1 | ![Static Badge](https://img.shields.io/badge/GitHub-SDv1-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 5. Stable Diffusion v2 | ![Static Badge](https://img.shields.io/badge/GitHub-SDv2-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 6. Stable Diffusion XL | ![Static Badge](https://img.shields.io/badge/GitHub-SDXL-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 7. Stable Diffusion v3 | ![Static Badge](https://img.shields.io/badge/GitHub-SDv3-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |

### 第 5 章: 拡散モデルによる画像生成技術の応用

| Section | GitHub | Colab |
|---------|---------|------|
| 1. パーソナライズされた画像生成| ![Static Badge](https://img.shields.io/badge/GitHub-Textual--Inversion-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-DreamBooth-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 2. 制御可能な画像生成 | ![Static Badge](https://img.shields.io/badge/GitHub-Attend--and--Excite-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-ControlNet-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 3. 拡散モデルによる画像編集 | ![Static Badge](https://img.shields.io/badge/GitHub-Prompt--to--Prompt-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-InstructPix2Pix-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-Paint--by--Example-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 4. 画像生成モデルの学習および推論の効率化 | ![Static Badge](https://img.shields.io/badge/GitHub-LoRA-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-LCM-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 5. 学習済み拡散モデルの効果的な拡張 | ![Static Badge](https://img.shields.io/badge/GitHub-GLIGEN-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-SDXL--turbo-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| 6. 生成画像の倫理・公平性 | ![Static Badge](https://img.shields.io/badge/GitHub-SLD-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |
| | ![Static Badge](https://img.shields.io/badge/GitHub-TIME-black?logo=github) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |

### 第 6 章: 画像生成の今後

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 拡散モデルの発展に伴う議論 | --- | --- |
| 2. 拡散モデルによる画像生成の倫理 | --- | --- |
| 3. 画像生成にとどまらない拡散モデルのの進化と今後 | --- | --- |

## 💬 コラム

- 📕 書籍 『Pythonで学ぶ画像生成』コラム補足記事｜Pythonで学ぶ画像生成｜note https://note.com/py_img_gen/m/m84877e9f9649 

## 🔗 関連リンク

- 🐍 Pythonで学ぶ画像生成 ランディングページ: https://py-img-gen.github.io/

## ❓ 疑問点・修正点

疑問点や修正点は以下の Issue にて管理しています。不明点などございましたら以下を確認し、解決方法が見つからない場合には新しく Issue を作成してください。
> https://github.com/py-img-gen/python-image-generation/issues
