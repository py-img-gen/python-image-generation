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
# # PyTorchによる基本的な訓練の流れ

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-2_pytorch.ipynb)

# %% [markdown]
# ## 準備


# %%
# !pip install -qq py-img-gen

# %%
import pathlib

current_dir = pathlib.Path.cwd()
project_dir = current_dir / "data" / "pytorch-introduction"

project_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## PyTorch の導入
#
# - 参考: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# %% [markdown]
# ## データの扱い方

# %% [markdown]
# ### データのダウンロードと読み込み

# %%
from torchvision import datasets
from torchvision.transforms import ToTensor

data_dir = project_dir / "dataset"

train_data = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=ToTensor(),
)

# %% [markdown]
# ### データローダーの作成

# %%
from torch.utils.data import DataLoader

batch_size = 64

train_dataloader = DataLoader(
    train_data, batch_size=batch_size
)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size
)

# %% [markdown]
# ### データローダーを用いたミニバッチの作成

# %%
(Xs, ys) = next(iter(train_dataloader))
print(f"Shape of Xs [N, C, H, W]: {Xs.shape}")
print(f"Shape of ys: {ys.shape} {ys.dtype}")

# %% [markdown]
# ### 訓練データの可視化

# %%
from py_img_gen.datasets import get_fashion_mnist_classes

classes = get_fashion_mnist_classes()
print(classes)

# %%
import matplotlib.pyplot as plt

idx = 0
# shape (X): (1, 28, 28) -> (28, 28)
X, y = Xs[idx].squeeze(dim=0), ys[idx]

fig, ax = plt.subplots()
ax.imshow(X, cmap="gray")
ax.set_title(f"Class: {y} ({classes[y]})")

# %% [markdown]
# ## モデルの構築

# %% [markdown]
# ### デバイスの取得

# %%
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using {device} device")


# %% [markdown]
# ## モデルの定義


# %%
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


model = NeuralNetwork()
model = model.to(device)

# %% [markdown]
# ## モデルパラメータの最適化

# %%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# %% [markdown]
# ### モデルの訓練方法の定義


# %%
def train(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
) -> None:
    model.train()  # モデルを訓練モードに

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 予測誤差の計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # 逆伝播の実行
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 != 0:
            continue

        current = (batch + 1) * len(X)
        size = len(dataloader.dataset)
        print(
            f"loss: {loss.item():>7f} "
            f"[{current:>5d}/{size:>5d}]"
        )


# %% [markdown]
# ### モデルの評価方法の定義


# %%
@torch.no_grad()
def test(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (
            (pred.argmax(1) == y)
            .type(torch.float)
            .sum()
            .item()
        )
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )


# %% [markdown]
# ### モデルの訓練と評価の実行

# %%
epochs = 5

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# %% [markdown]
# ## モデルの保存

# %%
save_path = project_dir / "model.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved PyTorch Model State to {save_path}")

# %% [markdown]
# ## モデルの読み込み

# %%
model = NeuralNetwork().to(device)
model.load_state_dict(
    state_dict=torch.load(save_path, weights_only=True)
)

# %% [markdown]
# ## 学習済みモデルを用いた予測

# %% [markdown]
# ### 予測の実行

# %%
model.eval()

(x, y) = test_data[0]

with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = (
        classes[pred[0].argmax(0)],
        classes[y],
    )

print(f'Predicted: "{predicted}", Actual: "{actual}"')

# %% [markdown]
# ### 予測結果の可視化

# %%
x = x.cpu()
x = x.squeeze(dim=0)

fig, ax = plt.subplots()
ax.imshow(x, cmap="gray")
ax.set_title(f"Class: {y} ({classes[y]})")
