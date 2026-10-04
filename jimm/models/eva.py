"""EVA-family vision transformers in flax nnx, NHWC. Mirrors timm.models.eva.

One configurable transformer covers EVA/EVA-02, the RoPE ViTs (axial and
mixed), DINOv3, LingBot and Perception Encoder towers: patch embedding with
optional class/register tokens and absolute position embedding, 2D rotary
embeddings on the patch tokens (timm's "cat", "mixed" with learned per-layer
frequencies, or DINOv3 normalized-coordinate periods), EVA attention (q/v
biases, optional inner LayerNorm) or plain fused-QKV attention, GELU, SwiGLU
or GLU MLPs (optionally normalized), layer scale, pre- or post-norm blocks,
and token, average or attention-pool heads.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model
from ._eva_cfgs import EVA_CFGS

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _ln(dim, eps, *, rngs):
    return nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)


# ----------------------------------------------------------------------------- rotary embeddings


def _rope_cat_table(head_dim, grid, temperature, grid_offset, ref_feat_shape, grid_indexing):
    """timm RotaryEmbeddingCat (in_pixels=False): [sin | cos] of shape (H*W, 2 * head_dim)."""
    num_bands = head_dim // 4
    bands = 1.0 / temperature ** (np.arange(num_bands, dtype=np.float32) / num_bands)
    feat = list(grid)
    ref = list(ref_feat_shape) if ref_feat_shape is not None else None
    if grid_indexing == "xy":
        feat = [feat[1], feat[0]]
        ref = [ref[1], ref[0]] if ref is not None else None
    t = [np.arange(s, dtype=np.float32) + grid_offset for s in feat]
    if ref is not None:
        t = [x / f * r for x, f, r in zip(t, feat, ref)]
    mesh = np.meshgrid(*t, indexing=grid_indexing)
    pos = np.stack(mesh, axis=-1)[..., None] * bands  # (.., .., 2, bands)
    n = grid[0] * grid[1]
    sin = np.repeat(np.sin(pos).reshape(n, -1), 2, axis=-1)
    cos = np.repeat(np.cos(pos).reshape(n, -1), 2, axis=-1)
    return np.concatenate([sin, cos], axis=-1).astype(np.float32)


def _rope_dinov3_table(head_dim, grid, temperature, grid_indexing, rotate_half=True):
    """timm RotaryEmbeddingDinoV3: 0.5-centered coords in [-1, 1], periods temperature**(2i/(d/2))."""
    h, w = grid
    dim4 = head_dim // 4
    periods = temperature ** (2.0 * np.arange(dim4, dtype=np.float32) / (head_dim // 2))
    ch = (np.arange(0.5, h, dtype=np.float32)) / h
    cw = (np.arange(0.5, w, dtype=np.float32)) / w
    if grid_indexing == "xy":
        gw, gh = np.meshgrid(cw, ch, indexing="xy")
        coords = np.stack([gh, gw], axis=-1)
    else:
        coords = np.stack(np.meshgrid(ch, cw, indexing="ij"), axis=-1)
    coords = 2.0 * coords.reshape(-1, 2) - 1.0
    angles = (2 * math.pi * coords[:, :, None] / periods[None, None, :]).reshape(h * w, -1)
    angles = np.tile(angles, 2) if rotate_half else np.repeat(angles, 2, axis=-1)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)


class RotaryEmbeddingMixed(nnx.Module):
    """Learned per-layer, per-head 2D frequencies (timm RotaryEmbeddingMixed)."""

    def __init__(self, dim, depth, num_heads, grid, temperature=10.0, grid_indexing="ij", *, rngs):
        head_dim = dim // num_heads
        mag = 1.0 / temperature ** (np.arange(0, head_dim, 4, dtype=np.float32) / head_dim)
        angles = jax.random.uniform(rngs.params(), (depth, num_heads, 1)) * 2 * math.pi
        fx = jnp.concatenate([mag * jnp.cos(angles), mag * jnp.cos(angles + math.pi / 2)], axis=-1)
        fy = jnp.concatenate([mag * jnp.sin(angles), mag * jnp.sin(angles + math.pi / 2)], axis=-1)
        self.freqs = nnx.Param(jnp.stack([fx, fy], axis=0))
        shape = list(grid)
        if grid_indexing == "xy":
            shape = [shape[1], shape[0]]
        x_pos, y_pos = np.meshgrid(
            np.arange(shape[0], dtype=np.float32), np.arange(shape[1], dtype=np.float32),
            indexing=grid_indexing,
        )  # fmt: skip
        self.t_x = nnx.Variable(jnp.asarray(x_pos.reshape(-1)))
        self.t_y = nnx.Variable(jnp.asarray(y_pos.reshape(-1)))

    def __call__(self):
        """(depth, heads, N, 2 * head_dim) [sin | cos] tables."""
        f = self.freqs[...].astype(jnp.float32)
        combined = (
            self.t_x[...][:, None] * f[0][..., None, :]
            + self.t_y[...][:, None] * f[1][..., None, :]
        )
        sin = jnp.repeat(jnp.sin(combined), 2, axis=-1)
        cos = jnp.repeat(jnp.cos(combined), 2, axis=-1)
        return jnp.concatenate([sin, cos], axis=-1)


def _rot(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(x, emb, half):
    """x: (B, N, H, D); emb: (N, 2D) or (H, N, 2D)."""
    if emb.ndim == 3:
        emb = jnp.transpose(emb, (1, 0, 2))[None]  # (1, N, H, 2D)
    else:
        emb = emb[None, :, None, :]
    sin, cos = jnp.split(emb.astype(x.dtype), 2, axis=-1)
    return x * cos + (_rotate_half(x) if half else _rot(x)) * sin


# ----------------------------------------------------------------------------- blocks


class EvaAttention(nnx.Module):
    def __init__(
        self, dim, num_heads, qkv_bias, qkv_fused, num_prefix, scale_norm, rotate_half, eps,
        eva_bias=True, *, rngs,
    ):  # fmt: skip
        self.num_heads, self.num_prefix, self.rotate_half = num_heads, num_prefix, rotate_half
        self.eva_bias = eva_bias
        if qkv_fused:
            # EVA attention: bias-free fused qkv with separate q/v biases (k bias fixed at zero).
            self.qkv = nnx.Linear(
                dim, dim * 3, use_bias=not eva_bias and qkv_bias, **_LINEAR, rngs=rngs
            )
            has = eva_bias and qkv_bias
            self.q_bias = nnx.Param(jnp.zeros(dim)) if has else None
            self.v_bias = nnx.Param(jnp.zeros(dim)) if has else None
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv = self.q_bias = self.v_bias = None
            k_bias = qkv_bias and not eva_bias
            self.q_proj = nnx.Linear(dim, dim, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
            self.k_proj = nnx.Linear(dim, dim, use_bias=k_bias, **_LINEAR, rngs=rngs)
            self.v_proj = nnx.Linear(dim, dim, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.norm = _ln(dim, eps, rngs=rngs) if scale_norm else None
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)

    def __call__(self, x, rope=None):
        B, N, C = x.shape
        h = self.num_heads
        if self.qkv is not None:
            qkv = self.qkv(x)
            if self.q_bias is not None:
                bias = jnp.concatenate(
                    [self.q_bias[...], jnp.zeros_like(self.q_bias[...]), self.v_bias[...]]
                )
                qkv = qkv + bias.astype(qkv.dtype)
            qkv = qkv.reshape(B, N, 3, h, C // h)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            q = self.q_proj(x).reshape(B, N, h, -1)
            k = self.k_proj(x).reshape(B, N, h, -1)
            v = self.v_proj(x).reshape(B, N, h, -1)
        if rope is not None:
            p = self.num_prefix
            q = jnp.concatenate([q[:, :p], _apply_rope(q[:, p:], rope, self.rotate_half)], axis=1)
            k = jnp.concatenate([k[:, :p], _apply_rope(k[:, p:], rope, self.rotate_half)], axis=1)
        x = dot_product_attention(q, k, v).reshape(B, N, C)
        if self.norm is not None:
            x = self.norm(x)
        return self.proj(x)


class Mlp(nnx.Module):
    """GELU MLP, optionally normalizing the hidden activations (scale_mlp)."""

    def __init__(self, dim, hidden, norm_eps=None, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden, **_LINEAR, rngs=rngs)
        self.norm = _ln(hidden, norm_eps, rngs=rngs) if norm_eps is not None else None
        self.fc2 = nnx.Linear(hidden, dim, **_LINEAR, rngs=rngs)

    def __call__(self, x):
        x = gelu(self.fc1(x))
        if self.norm is not None:
            x = self.norm(x)
        return self.fc2(x)


class SwiGLU(nnx.Module):
    def __init__(self, dim, hidden, norm_eps=None, align_to=0, *, rngs):
        if align_to:
            hidden = hidden + (-hidden % align_to)
        self.fc1_g = nnx.Linear(dim, hidden, **_LINEAR, rngs=rngs)
        self.fc1_x = nnx.Linear(dim, hidden, **_LINEAR, rngs=rngs)
        self.norm = _ln(hidden, norm_eps, rngs=rngs) if norm_eps is not None else None
        self.fc2 = nnx.Linear(hidden, dim, **_LINEAR, rngs=rngs)

    def __call__(self, x):
        x = nnx.silu(self.fc1_g(x)) * self.fc1_x(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.fc2(x)


class GluMlp(nnx.Module):
    """fc1 to twice the width, SiLU on the first half gating the second (gate_last=False)."""

    def __init__(self, dim, hidden2, norm_eps=None, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden2, **_LINEAR, rngs=rngs)
        self.norm = _ln(hidden2 // 2, norm_eps, rngs=rngs) if norm_eps is not None else None
        self.fc2 = nnx.Linear(hidden2 // 2, dim, **_LINEAR, rngs=rngs)

    def __call__(self, x):
        x1, x2 = jnp.split(self.fc1(x), 2, axis=-1)
        x = nnx.silu(x1) * x2
        if self.norm is not None:
            x = self.norm(x)
        return self.fc2(x)


def _mlp(dim, mlp_ratio, swiglu, scale_mlp, align_to, eps, rngs):
    hidden = int(dim * mlp_ratio)
    norm_eps = eps if scale_mlp else None
    if swiglu:
        if scale_mlp or align_to:
            return SwiGLU(dim, hidden, norm_eps, align_to, rngs=rngs)
        return GluMlp(dim, hidden * 2, norm_eps, rngs=rngs)
    return Mlp(dim, hidden, norm_eps, rngs=rngs)


class EvaBlock(nnx.Module):
    def __init__(self, dim, num_heads, cfg, num_prefix, eps, dpr, post_norm, *, rngs):
        self.post_norm = post_norm
        self.norm1 = _ln(dim, eps, rngs=rngs)
        self.attn = EvaAttention(
            dim, num_heads, cfg["qkv_bias"], cfg["qkv_fused"], num_prefix,
            cfg["scale_attn_inner"], cfg["rope_rotate_half"], eps,
            eva_bias=cfg["attn_type"] != "rope", rngs=rngs,
        )  # fmt: skip
        iv = None if post_norm else cfg["init_values"]
        self.gamma_1 = nnx.Param(jnp.full((dim,), iv)) if iv is not None else None
        self.drop_path = DropPath(dpr, rngs=rngs)
        self.norm2 = _ln(dim, eps, rngs=rngs)
        self.mlp = _mlp(
            dim, cfg["mlp_ratio"], cfg["swiglu_mlp"], cfg["scale_mlp"], cfg["swiglu_align_to"],
            eps, rngs,
        )  # fmt: skip
        self.gamma_2 = nnx.Param(jnp.full((dim,), iv)) if iv is not None else None

    def __call__(self, x, rope=None):
        if self.post_norm:
            x = x + self.drop_path(self.norm1(self.attn(x, rope)))
            return x + self.drop_path(self.norm2(self.mlp(x)))
        y = self.attn(self.norm1(x), rope)
        x = x + self.drop_path(y if self.gamma_1 is None else self.gamma_1[...] * y)
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y if self.gamma_2 is None else self.gamma_2[...] * y)


class AttentionPoolLatent(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, eps, *, rngs):
        self.num_heads = num_heads
        self.latent = nnx.Param(
            nnx.initializers.truncated_normal(dim**-0.5)(rngs.params(), (1, 1, dim))
        )
        self.q = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        self.kv = nnx.Linear(dim, dim * 2, **_LINEAR, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        self.norm = _ln(dim, eps, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(jnp.broadcast_to(self.latent[...], (B, 1, C))).reshape(B, 1, h, C // h)
        kv = self.kv(x).reshape(B, N, 2, h, C // h)
        x = self.proj(dot_product_attention(q, kv[:, :, 0], kv[:, :, 1]).reshape(B, 1, C))
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


_DEFAULTS = dict(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
    qkv_fused=True, mlp_ratio=4.0, swiglu_mlp=False, swiglu_align_to=0, scale_mlp=False,
    scale_attn_inner=False, attn_type="eva", norm_eps=1e-6, init_values=None, class_token=True,
    num_reg_tokens=0, no_embed_class=False, use_abs_pos_emb=True, use_rot_pos_emb=False,
    rope_type="cat", rope_grid_offset=0.0, rope_grid_indexing="ij", rope_temperature=10000.0,
    rope_rotate_half=False, use_post_norm=False, use_pre_transformer_norm=False,
    use_post_transformer_norm=None, use_fc_norm=None, attn_pool_num_heads=None,
    attn_pool_mlp_ratio=None, dynamic_img_size=False, ref_feat_shape=None,
)  # fmt: skip


class Eva(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0,
        *, rngs, **overrides,
    ):  # fmt: skip
        cfg = {**_DEFAULTS, **overrides}
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        self.num_classes, self.global_pool = num_classes, global_pool
        dim, depth, heads, eps = cfg["embed_dim"], cfg["depth"], cfg["num_heads"], cfg["norm_eps"]
        self.num_features = dim
        self.num_prefix = (1 if cfg["class_token"] else 0) + cfg["num_reg_tokens"]
        self.no_embed_class = cfg["no_embed_class"]
        fc_norm = cfg["use_fc_norm"] if cfg["use_fc_norm"] is not None else global_pool == "avg"
        post_norm = cfg["use_post_transformer_norm"]
        post_norm = (not fc_norm) if post_norm is None else post_norm
        p = cfg["patch_size"]
        self.patch_embed = _PatchEmbed(
            in_chans, dim, p, not cfg["use_pre_transformer_norm"], rngs=rngs
        )
        grid = (cfg["img_size"] // p,) * 2
        n = grid[0] * grid[1]
        self.cls_token = (
            nnx.Param(_trunc(rngs.params(), (1, 1, dim))) if cfg["class_token"] else None
        )
        r = cfg["num_reg_tokens"]
        self.reg_token = nnx.Param(_trunc(rngs.params(), (1, r, dim))) if r else None
        npos = n if self.no_embed_class else n + self.num_prefix
        self.pos_embed = (
            nnx.Param(_trunc(rngs.params(), (1, npos, dim))) if cfg["use_abs_pos_emb"] else None
        )
        rope = rope_table = None
        if cfg["use_rot_pos_emb"]:
            hd = dim // heads
            if cfg["rope_type"] == "mixed":
                rope = RotaryEmbeddingMixed(
                    dim, depth, heads, grid, cfg["rope_temperature"], cfg["rope_grid_indexing"],
                    rngs=rngs,
                )  # fmt: skip
            elif cfg["rope_type"] == "dinov3":
                table = _rope_dinov3_table(
                    hd, grid, cfg["rope_temperature"], cfg["rope_grid_indexing"]
                )
                rope_table = nnx.Variable(jnp.asarray(table))
            else:
                table = _rope_cat_table(
                    hd, grid, cfg["rope_temperature"], cfg["rope_grid_offset"],
                    cfg["ref_feat_shape"], cfg["rope_grid_indexing"],
                )  # fmt: skip
                rope_table = nnx.Variable(jnp.asarray(table))
        self.rope, self.rope_table = rope, rope_table
        self.norm_pre = _ln(dim, eps, rngs=rngs) if cfg["use_pre_transformer_norm"] else None
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                EvaBlock(
                    dim, heads, cfg, self.num_prefix, eps, dpr[i], cfg["use_post_norm"], rngs=rngs
                )
                for i in range(depth)
            ]
        )
        self.norm = _ln(dim, eps, rngs=rngs) if post_norm else None
        self.attn_pool = (
            AttentionPoolLatent(
                dim,
                cfg["attn_pool_num_heads"] or heads,
                cfg["attn_pool_mlp_ratio"] or cfg["mlp_ratio"],
                eps,
                rngs=rngs,
            )  # fmt: skip
            if global_pool == "map"
            else None
        )
        self.fc_norm = _ln(dim, eps, rngs=rngs) if fc_norm else None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, **_LINEAR, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, C = x.shape
        prefix = []
        if self.cls_token is not None:
            prefix.append(jnp.broadcast_to(self.cls_token[...], (B, 1, C)))
        if self.reg_token is not None:
            prefix.append(jnp.broadcast_to(self.reg_token[...], (B, *self.reg_token.shape[1:])))
        if self.no_embed_class:
            if self.pos_embed is not None:
                x = x + self.pos_embed[...]
            x = jnp.concatenate(prefix + [x], axis=1) if prefix else x
        else:
            x = jnp.concatenate(prefix + [x], axis=1) if prefix else x
            if self.pos_embed is not None:
                x = x + self.pos_embed[...]
        if self.norm_pre is not None:
            x = self.norm_pre(x)
        ropes = self.rope() if self.rope is not None else None
        table = self.rope_table[...] if self.rope_table is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, ropes[i] if ropes is not None else table)
        return self.norm(x) if self.norm is not None else x

    def forward_head(self, x):
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "token":
            x = x[:, 0]
        elif self.global_pool in ("avg", "max", "avgmax"):
            t = x[:, self.num_prefix :]
            if self.global_pool == "avg":
                x = t.mean(axis=1)
            elif self.global_pool == "max":
                x = t.max(axis=1)
            else:
                x = 0.5 * (t.mean(axis=1) + t.max(axis=1))
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


class _PatchEmbed(nnx.Module):
    def __init__(self, in_chans, dim, patch_size, bias, *, rngs):
        p = (patch_size, patch_size)
        self.proj = nnx.Conv(in_chans, dim, p, strides=p, padding="VALID", use_bias=bias, rngs=rngs)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


def _make(name):
    overrides, ev = EVA_CFGS[name]
    overrides = {k: v for k, v in overrides.items() if k != "global_pool"}
    pool = EVA_CFGS[name][0].get("global_pool", "avg")

    def entry(**kwargs):
        kwargs.setdefault("global_pool", pool)
        model = Eva(**{**overrides, **kwargs})
        model.default_cfg = _cfg(fixed_input_size=True, **ev)
        return model

    entry.__name__ = name
    return entry


for _name in EVA_CFGS:
    register_model(_make(_name))
