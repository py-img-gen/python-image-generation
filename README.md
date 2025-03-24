# Pythonで学ぶ画像生成（機械学習実践シリーズ）

[![CI](https://github.com/py-img-gen/python-image-generation/actions/workflows/ci.yaml/badge.svg)](https://github.com/py-img-gen/python-image-generation/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Appach--2.0-blue)](https://github.com/py-img-gen/python-image-generation/blob/main/LICENSE)
![Python](https://img.shields.io/badge/🐍%20Python-3.10+-orange)
[![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.31.0-orange)](https://github.com/huggingface/diffusers)

<img align="right" width="30%" src="https://github.com/user-attachments/assets/41bf761b-b55c-49d9-a273-df34c68f4a4b" alt="Pythonで学ぶ画像生成">

本レポジトリではインプレス社より出版されている [北田 俊輔](https://shunk031.me/) 著 の機械学習シリーズ「[Pythonで学ぶ画像生成](https://book.impress.co.jp/books/1123101104)」で扱うソースコードを管理しています。
ソースコードは Jupyter Notebook 形式でまとめられており、Google Colab 等で実行することを想定しています。

ソースコードの解説は主に書籍内に記載されており、本レポジトリのソースコードは補助教材となっています。

> [!WARNING]
> 本書で使用する環境やライブラリの変更等に伴い、補助教材の内容を予告なく更新する場合があります。そのため、常に最新の情報を確認し、本文の内容を読み替えてください。

本書籍は「**画像生成の基礎から実践までを一冊に凝縮**」というテーマで各章が構成されています。
まず​「画像生成とは​何か」と​いう​基本を​解説し、​次に​画像生成を​支える​深層学習の​基礎を​押さえます。​その上で、​現在の​最先端技術である​拡散モデルと、​その​効率化・応用例と​して​Stable Diffusionなどを​詳しく​取り上げています。​最後には、​拡散モデルが​もたらす革新的な​可能性と​同時に、​技術の​制限や​倫理的な​課題にも​言及し、​将来の​さらなる​発展・応用に​向けた​展望を​示しています。

-  6章 + 各章末に実装に役立つコラム付き
-  Python・PyTorchで学ぶ画像生成の実装
-  Diffusersによる最先端技術の実践
-  画像生成を中心とした様々なタスクの解説を多数収録

## 📦 動作要件

本書のサンプルコードではでは以下の環境での動作を想定しています。

| 🐍 / 📦 | ドキュメント / レポジトリ                                | 最低要件    |
|-----------------|---------------------------------------------|-------------|
| 🐍 python       | https://docs.python.org/ja/3.10/             | 3.10 以上   |
| 📦 pytorch      | https://github.com/pytorch/pytorch           | 2.0 以上    |
| 📦 transformers | https://github.com/huggingface/transformers  | 4.48.0 以上 |
| 📦 diffusers    | https://github.com/huggingface/diffusers     | 0.31.0 以上 |
| 📦 py-img-gen   | https://github.com/py-img-gen/py-img-gen-lib | 0.1.0 以上  |

## 📕 書籍の内容と補助教材

Jupyter Notebook の補助教材があるセクションには `Open in Colab` のバッジを付与しています。バッジをクリックすると該当するノートブックを Colab で開くことができます。

> [!NOTE]
> 本レポジトリでは [jupytext](https://github.com/mwouts/jupytext) を使用して [`scripts/`](https://github.com/py-img-gen/python-image-generation/tree/main/.jupytext/scripts) 以下に各 Jupyter Notebook に対応するスクリプトを用意しています。合わせてご確認ください。

### 第 1 章: 画像生成とは？

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 画像生成の概要 | --- |  --- |
| 2. テキストからの画像生成 | [![Open In GitHub](https://img.shields.io/badge/GitHub-Text--to--Image-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/1-2_text-to-image-generation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/1-2_text-to-image-generation.ipynb) |
| 3. 画像生成技術の進歩による弊害 | --- | --- |

### 第 2 章: 深層学習の基礎知識

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 深層学習の概要 | --- | --- |
| 2. 深層学習の訓練と評価 | [![Open In GitHub](https://img.shields.io/badge/GitHub-PyTorch-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/2-2_pytorch.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-2_pytorch.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-Activation--Function-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/2-2_activation-function.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-2_activation-function.ipynb) |
| 3. 注意機構と Transformer モデル | [![Open In GitHub](https://img.shields.io/badge/GitHub-Transformer-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb) |

### 第 3 章: 拡散モデルの導入

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 生成モデル | --- | --- |
| 2. DDPM（ノイズ除去拡散確率モデル）| [![Open In GitHub](https://img.shields.io/badge/GitHub-DDPM-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/3-2-1_ddpm.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-2-1_ddpm.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-DDIM-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/3-2-2_ddim.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-2-2_ddim.ipynb) |
| 3. スコアベース生成モデル | [![Open In GitHub](https://img.shields.io/badge/GitHub-NCSN-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/3-3_ncsn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-3_ncsn.ipynb) |
| 4. 拡散モデルの生成品質の向上 | [![Open In GitHub](https://img.shields.io/badge/GitHub-CFG-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/3-4_classifier-free-guidance.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/3-4_classifier-free-guidance.ipynb) |

### 第 4 章: 潜在拡散モデルと Stable Diffusion

| Section | GitHub | Colab |
|---------|---------|------|
| 1. LDM（潜在拡散確率モデル） | --- | --- |
| 2. CLIP | [![Open In GitHub](https://img.shields.io/badge/GitHub-CLIP-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/4-2_clip.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-2_clip.ipynb) |
| 3. Stable Diffusion を構成する要素 | [![Open In GitHub](https://img.shields.io/badge/GitHub-SD--Components-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/4-3_stable-diffusion_components.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-3_stable-diffusion_components.ipynb) |
| 4. Stable Diffusion v1 | [![Open In GitHub](https://img.shields.io/badge/GitHub-SDv1-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/4-4_stable-diffusion-v1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-4_stable-diffusion-v1.ipynb) |
| 5. Stable Diffusion v2 | [![Open In GitHub](https://img.shields.io/badge/GitHub-SDv2-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/4-5_stable-diffusion-v2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-5_stable-diffusion-v2.ipynb) |
| 6. Stable Diffusion XL | [![Open In GitHub](https://img.shields.io/badge/GitHub-SDXL-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/4-6_stable-diffusion-xl.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-6_stable-diffusion-xl.ipynb) |
| 7. Stable Diffusion v3 | [![Open In GitHub](https://img.shields.io/badge/GitHub-SDv3-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/4-7_stable-diffusion-v3.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/4-7_stable-diffusion-v3.ipynb) |

### 第 5 章: 拡散モデルによる画像生成技術の応用

| Section | GitHub | Colab |
|---------|---------|------|
| 1. パーソナライズされた画像生成| [![Open In GitHub](https://img.shields.io/badge/GitHub-Textual--Inversion-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-1-1_textual-inversion.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-1-1_textual-inversion.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-DreamBooth-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-1-2_dreambooth.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-1-2_dreambooth.ipynb) |
| 2. 制御可能な画像生成 | [![Open In GitHub](https://img.shields.io/badge/GitHub-Attend--and--Excite-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-2-1_attend-and-excite.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-2-1_attend-and-excite.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-ControlNet-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-2-2_controlnet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-2-2_controlnet.ipynb) |
| 3. 拡散モデルによる画像編集 | [![Open In GitHub](https://img.shields.io/badge/GitHub-Prompt--to--Prompt-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-3-1_prompt-to-prompt.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-3-1_prompt-to-prompt.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-InstructPix2Pix-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-3-2_instruct-pix2pix.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-3-2_instruct-pix2pix.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-Paint--by--Example-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-3-3_paint-by-example.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-3-3_paint-by-example.ipynb) |
| 4. 画像生成モデルの学習および推論の効率化 | [![Open In GitHub](https://img.shields.io/badge/GitHub-LoRA-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-4-1_lora.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-4-1_lora.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-LCM-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-4-2_latent-consistency-model.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-4-2_latent-consistency-model.ipynb) |
| 5. 学習済み拡散モデルの効果的な拡張 | [![Open In GitHub](https://img.shields.io/badge/GitHub-GLIGEN-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-5-1_gligen.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-5-1_gligen.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-SDXL--turbo-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-5-2_sdxl-turbo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-5-2_sdxl-turbo.ipynb) |
| 6. 生成画像の倫理・公平性 | [![Open In GitHub](https://img.shields.io/badge/GitHub-SLD-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-6-1_safe-latent-diffusion.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-6-1_safe-latent-diffusion.ipynb) |
| | [![Open In GitHub](https://img.shields.io/badge/GitHub-TIME-black?logo=github)](https://github.com/py-img-gen/python-image-generation/blob/main/notebooks/5-6-2_text-to-image-model-editing.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/5-6-2_text-to-image-model-editing.ipynb) |

### 第 6 章: 画像生成の今後

| Section | GitHub | Colab |
|---------|---------|------|
| 1. 拡散モデルの発展に伴う議論 | --- | --- |
| 2. 拡散モデルによる画像生成の倫理 | --- | --- |
| 3. 画像生成にとどまらない拡散モデルのの進化と今後 | --- | --- |

### 💬 コラム

- [書籍 『Pythonで学ぶ画像生成』コラム補足記事｜Pythonで学ぶ画像生成｜note](https://note.com/py_img_gen/m/m84877e9f9649 )
    - 本書の各章末にPythonを用いた実装に役立つコラムを1ページ程度で掲載しています。コラム補足記事では更に内容を拡張して、具体的にどのように理想的な実装につなげていくかについて詳細の説明を記載しています。

## 🔗 関連リンク

- 📄 [Pythonで学ぶ画像生成 ランディングページ](https://py-img-gen.github.io/)
    - 本書用のランディングページです。関連情報を掲載しています。
- ⚙️ [py-img-gen/py-img-gen-lib: 🐍 A library for the book "Image Generation with Python"](https://github.com/py-img-gen/py-img-gen-lib )
    - 本書で使用するライブラリの GitHub レポジトリです。補助教材を実行する際に必要な依存ライブラリのインストールや、教材内の一部コードを簡略化するために使用していただけます。
- ⚙️ [py-img-gen/diffusers-ncsn: 🤗 diffusers implementation of Noise Conditional Score Network (NCSN)](https://github.com/py-img-gen/diffusers-ncsn )
    - 本書で説明する [NCSN](https://arxiv.org/abs/1907.05600) を diffusers パイプラインとして実装した GitHub レポジトリです。README に従うとパイプラインをダウンロードおよび実行することができます。
- ⚙️ [py-img-gen/diffusers-text-to-model-editing: 🤗 Fork of the diffusers implementation of "Text to Model Editing (TIME)"](https://github.com/py-img-gen/diffusers-text-to-model-editing)
    - 本書で説明する [TIME](https://arxiv.org/abs/2303.08084) について、diffusers オフィシャルで用意されているものを fork して最新版 diffusers で正しく動作するように修正実装した GitHub レポジトリです。
- ⚙️ [py-img-gen/huggingface-datasets_ukiyo-e-face-blip2-captions: 🤗 Ukiyo-e-face dataset with BLIP2 captions for huggingface datasets](https://github.com/py-img-gen/huggingface-datasets_ukiyo-e-face-blip2-captions )
    - 本書で紹介する LoRA の実装例で浮世絵スタイルの画像生成を実現する方法について紹介しています。その際に [Ukiyo-e-face データセット](https://www.justinpinkney.com/blog/2020/ukiyoe-dataset/)に対して、[BLIP2](https://arxiv.org/abs/2301.12597) を適用して画像とキャプションのペアデータセットを作成しています。データセットは [huggingface datasets 形式で公開](https://huggingface.co/datasets/py-img-gen/ukiyo-e-face-blip2-captions)しています。本レポジトリでは huggingface datasets を作成する際の loading script を公開しています。
- 🎞️ [画像生成AI入門：Pythonによる拡散モデルの理論と実践 リサーチサイエンティスト 北田俊輔 | Coloso.](https://coloso.jp/products/researchscientist-kitada-jp )
    - 著者が Coloso. というビデオオンデマンド型オンライン教育サービスにて公開している動画コンテンツです。本書と併せて学習することで、より深い理解を得ることができると考えています。

## 🗣️ 読者の声

### [【書評】 Pythonで学ぶ画像生成 機械学習実践シリーズ｜npaka](https://note.com/npaka/n/n349aa47fe957)

> 近年、生成AIの技術革新により「画像生成」の世界は急速に発展しています。そんな中、注目すべき一冊が『Pythonで学ぶ画像生成 機械学習実践シリーズ』です。本書は、深層学習を基盤とした画像生成について、基礎理論から実践的なPython実装例までを丁寧に解説しており、画像生成の全貌に迫る実践的な指南書と言えるでしょう。

### [Pythonで学ぶ画像生成の書評｜あるふ](https://note.com/alfredplpl/n/n2809e615af4a)

> 画像生成の学びたい人にまず手渡したい本がやっとできたという気持ちになっています。著者の北田さんお疲れ様でした。（僕は気合が足りなくてかけませんでした・・・

### [この本から勉強できる人が羨ましいです。｜asap](https://x.com/asap2650/status/1903980641508831503)

> 前半は初学者でも分かりやすく、図や丁寧なコードで分かりやすく説明されており、読みやすいです。
> 図に関しても、ただ論文の図を貼っただけみたいなものではなく、今回のために分かりやすく作っていたり、注釈に理解できるように丁寧に解説が書いてあり、感動しました。
> 後半以降は、最近流行りの「画像編集」の手法や、プロンプト忠実性を高める手法、一貫性モデルなど、研究的な話も読めます。
> 内容的には少し難しくなるはずですが、とても分かりやすく簡潔に書かれているため、「手法の凄さ」と「設計の意図」がスッと入ってくるようになっており、無駄がないです。

## ❓ 疑問点・修正点

疑問点や修正点は以下の Issue にて管理しています。不明点などございましたら以下を確認し、解決方法が見つからない場合には新しく Issue を作成してください。

> https://github.com/py-img-gen/python-image-generation/issues

疑問点は他の読者にとっても疑問になり得ます。積極的な Issue の活用により、他の読者にとっても有益な知識の共有が進み、コミュニティ全体での問題解決が効率的になります。どうぞよろしくお願いします！
