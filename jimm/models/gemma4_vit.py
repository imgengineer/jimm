"""Gemma 4 vision tower in flax nnx, NHWC. Mirrors timm.models.gemma4_vit for image inputs.

Pixels in [0, 1] are patchified (channel-major within a patch), mapped to
[-1, 1] and projected without bias; learned per-axis position tables
(column, row) are added. Blocks use sandwich RMSNorm (before and after both
attention and the tanh-GELU gated MLP), q/k RMSNorm and a parameter-free v
RMSNorm, 2D rotary embeddings (half the head dims rotate with the column,
half with the row) and unscaled attention. Bias-free projections can clamp
their inputs and outputs. The encoder ("_enc") average-pools 3x3 patch
neighbourhoods by position, scales by sqrt(dim) and optionally standardizes;
the classifier averages all tokens, applies a parameter-free RMSNorm and a
linear head. Pre-patchified NaFlex inputs with padding are not supported.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)


def _gelu(x):
    return jax.nn.gelu(x, approximate=True)


class RmsNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        self.scale = nnx.Param(jnp.ones(dim)) if affine else None
        self.eps = eps

    def __call__(self, x):
        acc = jnp.promote_types(x.dtype, jnp.float32)
        var = jnp.mean(jnp.square(x.astype(acc)), axis=-1, keepdims=True)
        y = (x.astype(acc) * jax.lax.rsqrt(var + self.eps)).astype(x.dtype)
        return y * self.scale[...].astype(x.dtype) if self.scale is not None else y


class ClippableLinear(nnx.Module):
    def __init__(self, din, dout, use_clipped=False, *, rngs):
        self.linear = nnx.Linear(din, dout, use_bias=False, kernel_init=_trunc, rngs=rngs)
        self.use_clipped = use_clipped
        if use_clipped:  # checkpoint-provided bounds; infinite until loaded
            self.input_min = nnx.Variable(jnp.asarray(-jnp.inf))
            self.input_max = nnx.Variable(jnp.asarray(jnp.inf))
            self.output_min = nnx.Variable(jnp.asarray(-jnp.inf))
            self.output_max = nnx.Variable(jnp.asarray(jnp.inf))

    def __call__(self, x):
        if self.use_clipped:
            x = jnp.clip(x, self.input_min[...], self.input_max[...])
        x = self.linear(x)
        if self.use_clipped:
            x = jnp.clip(x, self.output_min[...], self.output_max[...])
        return x


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _rope_2d(x, cos, sin):
    """Rotate the first half of the head dims by the column, the second by the row."""
    half = 2 * (x.shape[-1] // 4)
    parts = []
    for k in range(2):
        sl = slice(k * half, (k + 1) * half)
        c, s = cos[:, :, None, sl], sin[:, :, None, sl]
        parts.append(x[..., sl] * c + _rotate_half(x[..., sl]) * s)
    return jnp.concatenate(parts, axis=-1)


def _rope_tables(position_ids, head_dim, theta, dtype):
    acc = jnp.promote_types(dtype, jnp.float32)
    spatial = head_dim // 2
    inv_freq = 1.0 / theta ** (jnp.arange(0, spatial, 2, dtype=acc) / spatial)
    cos, sin = [], []
    for i in range(2):
        freqs = position_ids[..., i, None].astype(acc) * inv_freq
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos.append(jnp.cos(emb))
        sin.append(jnp.sin(emb))
    return jnp.concatenate(cos, axis=-1), jnp.concatenate(sin, axis=-1)


class Gemma4PatchEmbed(nnx.Module):
    def __init__(self, patch_size, in_chans, embed_dim, position_embedding_size, *, rngs):
        self.patch_size = patch_size
        self.input_proj = nnx.Linear(
            in_chans * patch_size**2, embed_dim, use_bias=False, kernel_init=_trunc, rngs=rngs
        )
        self.position_embedding_table = nnx.Param(
            _trunc(rngs.params(), (2, position_embedding_size, embed_dim))
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        x = x.reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 5, 2, 4).reshape(B, gh * gw, -1)
        x = self.input_proj(2 * (x - 0.5))
        rows, cols = jnp.meshgrid(jnp.arange(gh), jnp.arange(gw), indexing="ij")
        position_ids = jnp.stack([cols.ravel(), rows.ravel()], axis=-1)  # (x, y) per patch
        table = self.position_embedding_table[...]
        x = x + table[0][position_ids[:, 0]] + table[1][position_ids[:, 1]]
        return x, jnp.broadcast_to(position_ids, (B, gh * gw, 2)), (gh, gw)


class Gemma4Attention(nnx.Module):
    def __init__(self, dim, num_heads, head_dim, clipped, eps, *, rngs):
        self.num_heads, self.head_dim = num_heads, head_dim
        inner = num_heads * head_dim
        self.q_proj = ClippableLinear(dim, inner, clipped, rngs=rngs)
        self.k_proj = ClippableLinear(dim, inner, clipped, rngs=rngs)
        self.v_proj = ClippableLinear(dim, inner, clipped, rngs=rngs)
        self.o_proj = ClippableLinear(inner, dim, clipped, rngs=rngs)
        self.q_norm = RmsNorm(head_dim, eps)
        self.k_norm = RmsNorm(head_dim, eps)
        self.v_norm = RmsNorm(head_dim, eps, affine=False)

    def __call__(self, x, cos, sin):
        B, N, _ = x.shape
        shape = (B, N, self.num_heads, self.head_dim)
        q = _rope_2d(self.q_norm(self.q_proj(x).reshape(shape)), cos, sin)
        k = _rope_2d(self.k_norm(self.k_proj(x).reshape(shape)), cos, sin)
        v = self.v_norm(self.v_proj(x).reshape(shape))
        # Unscaled attention: fold the 1/sqrt(d) that dot_product_attention applies into q.
        q = q * jnp.asarray(self.head_dim**0.5, q.dtype)
        out = dot_product_attention(q, k, v).reshape(B, N, -1)
        return self.o_proj(out)


class Gemma4GatedMlp(nnx.Module):
    def __init__(self, dim, hidden, clipped, *, rngs):
        self.gate_proj = ClippableLinear(dim, hidden, clipped, rngs=rngs)
        self.up_proj = ClippableLinear(dim, hidden, clipped, rngs=rngs)
        self.down_proj = ClippableLinear(hidden, dim, clipped, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(_gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma4Block(nnx.Module):
    def __init__(self, dim, num_heads, head_dim, hidden, clipped, eps, drop_path, *, rngs):
        self.norm1 = RmsNorm(dim, eps)
        self.attn = Gemma4Attention(dim, num_heads, head_dim, clipped, eps, rngs=rngs)
        self.norm2 = RmsNorm(dim, eps)
        self.norm3 = RmsNorm(dim, eps)
        self.mlp = Gemma4GatedMlp(dim, hidden, clipped, rngs=rngs)
        self.norm4 = RmsNorm(dim, eps)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x, cos, sin):
        x = x + self.drop_path(self.norm2(self.attn(self.norm1(x), cos, sin)))
        return x + self.drop_path(self.norm4(self.mlp(self.norm3(x))))


class Gemma4VitEncoder(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        global_pool="soft",
        embed_dim=768,
        depth=16,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        norm_eps=1e-6,
        rope_theta=100.0,
        position_embedding_size=10240,
        pooling_kernel_size=3,
        standardize=False,
        use_clipped_linears=False,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        assert global_pool in ("soft", "avg", "none", "")
        self.global_pool, self.num_classes, self.head = global_pool, 0, None
        self.num_features = embed_dim
        self.head_dim, self.rope_theta, self.pool_k = head_dim, rope_theta, pooling_kernel_size
        self.patch_embed = Gemma4PatchEmbed(
            patch_size, in_chans, embed_dim, position_embedding_size, rngs=rngs
        )
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Gemma4Block(
                    embed_dim,
                    num_heads,
                    head_dim,
                    intermediate_size,
                    use_clipped_linears,
                    norm_eps,
                    dpr[i],
                    rngs=rngs,
                )  # fmt: skip
                for i in range(depth)
            ]
        )
        if standardize:
            self.std_bias = nnx.Variable(jnp.zeros(embed_dim))
            self.std_scale = nnx.Variable(jnp.ones(embed_dim))
        else:
            self.std_bias = self.std_scale = None

    def reset_classifier(self, num_classes, global_pool=None):
        assert num_classes == 0, "the encoder has no classifier"

    def forward_features(self, x):
        x, position_ids, grid = self.patch_embed(x)
        cos, sin = _rope_tables(position_ids, self.head_dim, self.rope_theta, x.dtype)
        cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x, grid

    def _soft_pool(self, x, grid):
        """Average each k x k patch neighbourhood (row-major output) and scale by sqrt(dim)."""
        B, _, D = x.shape
        (gh, gw), k = grid, self.pool_k
        acc = jnp.promote_types(x.dtype, jnp.float32)  # timm pools in float32
        pooled = x.astype(acc).reshape(B, gh // k, k, gw // k, k, D).mean(axis=(2, 4))
        x = pooled.reshape(B, -1, D).astype(x.dtype) * jnp.asarray(D**0.5, x.dtype)
        if self.std_bias is not None:
            x = (x - self.std_bias[...]) * self.std_scale[...]
        return x

    def forward_head(self, x):
        return x

    def __call__(self, x):
        x, grid = self.forward_features(x)
        if self.global_pool == "soft":
            return self._soft_pool(x, grid)
        if self.global_pool == "avg":
            return x.mean(axis=1)
        return x


class Gemma4VitClassifier(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        num_classes=1000,
        global_pool="avg",
        final_norm=True,
        drop_rate=0.0,
        norm_eps=1e-6,
        *,
        rngs,
        **encoder_kwargs,
    ):
        assert global_pool in ("avg", "none", "")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.encoder = Gemma4VitEncoder(
            global_pool="", norm_eps=norm_eps, rngs=rngs, **encoder_kwargs
        )
        self.num_features = self.encoder.num_features
        self.norm = RmsNorm(self.num_features, norm_eps, affine=False) if final_norm else None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, kernel_init=_trunc, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("avg", "none", "")
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        return self.encoder.forward_features(x)[0]

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(axis=1)
        if self.norm is not None:
            x = self.norm(x)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_ARCHS = {
    "167m": dict(
        embed_dim=768,
        depth=16,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        use_clipped_linears=True,
    ),  # fmt: skip
    "570m": dict(
        embed_dim=1152,
        depth=27,
        num_heads=16,
        head_dim=72,
        intermediate_size=4304,
        standardize=True,
    ),  # fmt: skip
}


def _make(name):
    arch = _ARCHS[name.split("_")[2]]
    encoder = name.endswith("_enc")

    def entry(**kwargs):
        if encoder:
            kwargs.pop("num_classes", None)
            model = Gemma4VitEncoder(**{**arch, **kwargs})
        else:
            model = Gemma4VitClassifier(**{**arch, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, 768, 768),
            min_input_size=(3, 96, 96),
            crop_pct=1.0,
            interpolation="bicubic",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            num_classes=0,
        )
        return model

    entry.__name__ = name
    return entry


for _name in ("gemma4_vit_167m", "gemma4_vit_167m_enc", "gemma4_vit_570m", "gemma4_vit_570m_enc"):
    register_model(_make(_name))
