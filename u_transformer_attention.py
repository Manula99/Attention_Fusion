import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks import PatchEmbeddingBlock
from torch.nn.attention import SDPBackend, sdpa_kernel


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention module with kv caching. Supports padded or nested tensors.

    Args:
        embed_tot_dim (int): total embedded dimension size; each head has embed_tot_dim // num_heads
        q_dim (int): query embedding dimension
        v_dim (int): value embedding dimension
        k_dim (int): key embedding dimension
        num_heads (int): number of heads for attention block
        qkv_bias (bool) : whether to add bias (learn additional bias), defaults to False
    """
    def __init__(self, embed_tot_dim, q_dim, k_dim, v_dim, num_heads, dropout=0, qkv_bias=False):
        super().__init__()
        self._qkv_same_embed_dim = q_dim == k_dim and q_dim == v_dim
        if self._qkv_same_embed_dim:
            self.qkv = nn.Linear(q_dim, embed_tot_dim * 3, bias=qkv_bias)
        else:
            self.query_proj = nn.Linear(in_features=q_dim, out_features=embed_tot_dim, bias=qkv_bias)
            self.key_proj = nn.Linear(in_features=k_dim, out_features=embed_tot_dim, bias=qkv_bias)
            self.val_proj = nn.Linear(in_features=v_dim, out_features=embed_tot_dim, bias=qkv_bias)

        d_out = q_dim
        self.out_proj = nn.Linear(in_features=embed_tot_dim, out_features=d_out, bias=qkv_bias)

        assert embed_tot_dim % num_heads == 0, "embedding/channel dim is not divisible by num_heads!"

        self.num_heads = num_heads
        self.head_dim = embed_tot_dim // num_heads
        self.bias = qkv_bias
        self.dropout = dropout

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_model = x.size()
        return x.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
        use_cache=True,
    ) -> torch.Tensor:
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.qkv(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.qkv.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.qkv.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query = F.linear(query, q_weight, q_bias)
                key = F.linear(key, k_weight, k_bias)
                value = F.linear(value, v_weight, v_bias)
        else:
            query = self.query_proj(query)
            key = self.key_proj(key)
            value = self.val_proj(value)

        query = query.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        key = key.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        value = value.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = key, value
            else:
                self.cache_k = torch.cat([self.cache_k, key], dim=1)
                self.cache_v = torch.cat([self.cache_v, value], dim=1)
            key, value = self.cache_k, self.cache_v

        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout, is_causal=is_causal
            )

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        return self.out_proj(attn_output)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class MHSA(nn.Module):
    def __init__(self, channel, spatial_dims, patch_size, hidden, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbeddingBlock(channel, spatial_dims, patch_size, hidden, num_heads=num_heads)
        self.attn = MultiHeadAttention(
            embed_tot_dim=channel,
            q_dim=channel,
            k_dim=channel,
            v_dim=channel,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, d = x.size()
        x = self.patch_embed(x)
        attn_output = self.attn(x, x, x)
        x = attn_output.transpose(-1, -2).reshape(b, c, h, w, d)
        return x


class MHCA(nn.Module):
    def __init__(
        self,
        channelY,
        channelS,
        spat_dimS,
        spat_dimY,
        num_heads=8,
    ):
        super().__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(channelS, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS),
            nn.LeakyReLU(inplace=True),
        )
        self.Yconv = nn.Sequential(
            nn.Conv3d(channelY, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS),
            nn.LeakyReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(channelY, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True),
        )
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv3d(channelY, channelS, kernel_size=1),
            nn.BatchNorm3d(channelS),
            nn.LeakyReLU(inplace=True),
        )

        self.Spe = PatchEmbeddingBlock(channelS, spat_dimS, (4, 4, 4), channelY, num_heads)
        self.Ype = PatchEmbeddingBlock(channelY, spat_dimY, (2, 2, 2), channelY, num_heads)
        self.attn = MultiHeadAttention(
            embed_tot_dim=channelY,
            q_dim=channelY,
            k_dim=channelY,
            v_dim=channelY,
            num_heads=num_heads,
        )

    def forward(self, Y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        Sb, Sc, Sh, Sw, Sd = S.size()
        Yb, Yc, Yh, Yw, Yd = Y.size()
        S1 = self.Spe(S)
        Y1 = self.Ype(Y)
        Y2 = self.Yconv2(Y)
        print("attention input: ", Y1.size(), S1.size())
        attn_output = self.attn(Y1, S1, S1)
        print("initial attn output size", attn_output.size())
        attn_output = attn_output.permute(0, 2, 1).reshape(Yb, Yc, Yh // 2, Yw // 2, Yd // 2)
        print("attn output after reshape", attn_output.size())
        Z = self.conv(attn_output)
        print("MHCA conv done")
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z
