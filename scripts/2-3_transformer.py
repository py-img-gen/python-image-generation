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
# # PyTorch による Transformer の実装

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py-img-gen/python-image-generation/blob/main/notebooks/2-3_transformer.ipynb)

# %%
# !pip install -qq py-img-gen

# %% [markdown]
# ## 準備

# %%
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# %% [markdown]
# ## Transformer の実装例


# %%
class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoders: int = 6,
        num_decoders: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            d_model, num_heads, num_encoders
        )
        self.decoder = Decoder(
            d_model, num_heads, num_decoders
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(
            tgt, enc_out, src_mask, tgt_mask
        )
        return dec_out


# %% [markdown]
# ## Transformer Encoder の実装例


# %%
class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_encoders: int,
    ) -> None:
        super().__init__()
        layers = [
            EncoderLayer(d_model, num_heads)
            for _ in range(num_encoders)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        return output


# %% [markdown]
# ## Transformer Decoder の実装例


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_decoders: int,
    ) -> None:
        super().__init__()
        layers = [
            DecoderLayer(d_model, num_heads)
            for _ in range(num_decoders)
        ]
        self.dec_layers = nn.ModuleList(layers)

    def forward(
        self,
        tgt: torch.Tensor,
        enc: torch.Tensor,
        tgt_mask: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.dec_layers:
            output = layer(output, enc, tgt_mask, enc_mask)
        return output


# %% [markdown]
# ## Transformer Encoder レイヤーの実装例


# %%
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mha = MultiheadAttention(
            d_model, num_heads, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        x = src
        x = x * self.mha(q=x, k=x, v=x, mask=src_mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x


# %% [markdown]
# ## Multi-head Attention の実装例


# %%
class MultiheadAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        attn_output_size = self.d_model // self.num_heads
        attns = [
            SelfAttention(d_model, attn_output_size)
            for _ in range(self.num_heads)
        ]
        self.attentions = nn.ModuleList(attns)
        self.output = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                layer(q, k, v, mask)
                for layer in self.attentions
            ],
            dim=-1,
        )
        x = self.output(x)
        return x


# %% [markdown]
# ## Self-Attention の実装例


# %%
class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, output_size)
        self.k = nn.Linear(d_model, output_size)
        self.v = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = q.size(dim=0)
        tgt_len = q.size(dim=1)
        seq_len = k.size(dim=1)
        q, k, v = self.q(q), self.k(v), self.v(v)
        dim_k = k.size(dim=-1)
        scores = torch.bmm(
            q, k.transpose(1, 2)
        ) / torch.sqrt(dim_k)

        if mask is not None:
            mask = mask[:, None, :]
            expanded_mask = mask.expand(
                bs, tgt_len, seq_len
            )
            scores = scores.masked_fill(
                expanded_mask == 0, -float("Inf")
            )

        weights = F.softmax(scores, dim=-1)
        outputs = torch.bmm(weights, v)
        return outputs


# %% [markdown]
# ## Transformer Decoder レイヤーの実装例


# %%
class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.masked_mha = MultiheadAttention(
            d_model, num_heads, dropout
        )
        self.mha = MultiheadAttention(
            d_model, num_heads, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.masked_mha_norm = nn.LayerNorm(d_model)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        enc: torch.Tensor,
        tgt_mask: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = tgt
        x = x + self.masked_mha(
            q=x, k=x, v=x, mask=tgt_mask
        )
        x = self.masked_mha_norm(x)
        x = x + self.mha(q=x, k=enc, v=enc, mask=enc_mask)
        x = self.mha_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x
