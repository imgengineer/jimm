"""Vision Transformer in flax nnx, NHWC input. Mirrors timm.models.vision_transformer.

One configurable model covers timm's VisionTransformer registry (and DeiT's distilled
variant): class/register tokens, learned position embeddings with or without the prefix
tokens, LayerNorm or RMSNorm, GELU/QuickGELU/SiLU MLPs, SwiGLU and packed GLU MLPs, QK and
scale norms, layer scale, residual-post-norm, parallel and parallel-scaling blocks,
differential attention, and token/average/attention-pool (MAP) heads.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model
from ._vit_cfgs import VIT_CFGS


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


_ACTS = {
    "gelu": gelu,
    "quick_gelu": quick_gelu,
    "gelu_tanh": lambda x: nnx.gelu(x, approximate=True),
    "silu": nnx.silu,
}


def _norm(kind, dim, eps, *, rngs):
    if kind == "rmsnorm":
        return nnx.RMSNorm(dim, epsilon=eps, rngs=rngs)
    return nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)


class LayerScale(nnx.Module):
    def __init__(self, dim, init_values):
        self.gamma = nnx.Param(jnp.full((dim,), init_values, jnp.float32))

    def __call__(self, x):
        return x * self.gamma[...]


class Attention(nnx.Module):
    """timm ``Attention``: fused qkv, optional per-head QK norm and pre-projection norm."""

    def __init__(
        self, dim, num_heads=8, qkv_bias=True, drop=0.0, *, qk_norm=False, scale_norm=False,
        proj_bias=True, norm=None, rngs,
    ):  # fmt: skip
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        norm = norm or (lambda d: _norm("layernorm", d, 1e-6, rngs=rngs))
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.norm = norm(dim) if scale_norm else None
        self.proj = nnx.Linear(dim, dim, use_bias=proj_bias, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        x = dot_product_attention(q, k, v).reshape(B, N, C)
        if self.norm is not None:
            x = self.norm(x)
        return self.drop(self.proj(x))


def _diff_lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def _init_diff(module, head_dim, depth, *, rngs):
    """Adds differential-attention parameters: ``lambda = exp(q1.k1) - exp(q2.k2) + init``."""
    init = nnx.initializers.normal(0.1)
    for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
        setattr(module, name, nnx.Param(init(rngs.params(), (head_dim,))))
    module.lambda_init = _diff_lambda_init(depth)
    module.sub_norm = nnx.RMSNorm(2 * head_dim, epsilon=1e-5, rngs=rngs)


def _diff_attend(module, q, k, v):
    """``q``/``k``: (B, N, 2H, d); ``v``: (B, N, H, 2d) -> (B, N, H * 2d)."""
    lam1 = jnp.exp(jnp.sum(module.lambda_q1[...] * module.lambda_k1[...]))
    lam2 = jnp.exp(jnp.sum(module.lambda_q2[...] * module.lambda_k2[...]))
    lam = lam1 - lam2 + module.lambda_init
    a1 = dot_product_attention(q[:, :, 0::2], k[:, :, 0::2], v)
    a2 = dot_product_attention(q[:, :, 1::2], k[:, :, 1::2], v)
    x = module.sub_norm(a1 - lam * a2) * (1 - module.lambda_init)
    return x.reshape(*x.shape[:2], -1)


class DiffAttention(nnx.Module):
    """timm ``DiffAttention``: difference of two softmax maps over split query/key heads."""

    def __init__(
        self, dim, num_heads, qkv_bias, *, qk_norm, scale_norm, proj_bias, norm, depth, rngs
    ):
        self.num_heads, self.head_dim = num_heads, dim // num_heads // 2
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        _init_diff(self, self.head_dim, depth, rngs=rngs)
        self.norm = norm(dim) if scale_norm else None
        self.proj = nnx.Linear(dim, dim, use_bias=proj_bias, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, N, 2 * self.num_heads, self.head_dim)
        k = k.reshape(B, N, 2 * self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        x = _diff_attend(self, q, k, v.reshape(B, N, self.num_heads, -1))
        if self.norm is not None:
            x = self.norm(x)
        return self.proj(x)


class Mlp(nnx.Module):
    """timm ``Mlp`` / ``SwiGLU`` / ``GluMlp`` (``kind`` mlp, swiglu or glu)."""

    def __init__(self, dim, hidden, act, kind="mlp", norm=None, bias=True, drop=0.0, *, rngs):
        self.act, self.kind = act, kind
        if kind == "swiglu":
            self.fc1_g = nnx.Linear(dim, hidden, use_bias=bias, rngs=rngs)
            self.fc1_x = nnx.Linear(dim, hidden, use_bias=bias, rngs=rngs)
        else:
            self.fc1 = nnx.Linear(dim, hidden, use_bias=bias, rngs=rngs)
        out_in = hidden // 2 if kind == "glu" else hidden
        self.norm = norm(out_in) if norm is not None else None
        self.fc2 = nnx.Linear(out_in, dim, use_bias=bias, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        if self.kind == "swiglu":
            x = self.act(self.fc1_g(x)) * self.fc1_x(x)
        elif self.kind == "glu":
            x1, x2 = jnp.split(self.fc1(x), 2, axis=-1)
            x = self.act(x1) * x2
        else:
            x = self.act(self.fc1(x))
        x = self.drop(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.drop(self.fc2(x))


def _block_parts(dim, num_heads, mlp_ratio, qkv_bias, drop, o, depth, *, rngs):
    """Builds the (attention, mlp) pair shared by the sequential block types."""
    norm = o["norm"]
    if o["attn_layer"] == "diff":
        attn = DiffAttention(
            dim, num_heads, qkv_bias, qk_norm=o["qk_norm"], scale_norm=o["scale_attn_norm"],
            proj_bias=o["proj_bias"], norm=norm, depth=depth, rngs=rngs,
        )  # fmt: skip
    else:
        attn = Attention(
            dim, num_heads, qkv_bias, drop, qk_norm=o["qk_norm"], scale_norm=o["scale_attn_norm"],
            proj_bias=o["proj_bias"], norm=norm, rngs=rngs,
        )  # fmt: skip
    mlp = Mlp(
        dim, int(dim * mlp_ratio), o["act"], o["mlp_layer"],
        norm if o["scale_mlp_norm"] else None, o["proj_bias"], drop, rngs=rngs,
    )  # fmt: skip
    return attn, mlp


def _block_opts(opts, rngs):
    o = dict(
        qk_norm=False, scale_attn_norm=False, scale_mlp_norm=False, proj_bias=True, act=gelu,
        mlp_layer="mlp", attn_layer="", norm_layer="layernorm", norm_eps=1e-6,
    )  # fmt: skip
    o.update(opts)
    o["norm"] = lambda d: _norm(o["norm_layer"], d, o["norm_eps"], rngs=rngs)
    return o


class Block(nnx.Module):
    """Pre-norm transformer block with optional layer scale (``ls1``/``ls2``)."""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0,
        init_values=None, *, depth=0, rngs, **opts,
    ):  # fmt: skip
        o = _block_opts(opts, rngs)
        self.norm1 = o["norm"](dim)
        self.attn, self.mlp = _block_parts(
            dim, num_heads, mlp_ratio, qkv_bias, drop, o, depth, rngs=rngs
        )
        self.norm2 = o["norm"](dim)
        self.ls1 = LayerScale(dim, init_values) if init_values else None
        self.ls2 = LayerScale(dim, init_values) if init_values else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y if self.ls1 is None else self.ls1(y))
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y if self.ls2 is None else self.ls2(y))


class ResPostBlock(nnx.Module):
    """``x + norm(f(x))``; ``init_values`` initializes the norm weights."""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0,
        init_values=None, *, depth=0, rngs, **opts,
    ):  # fmt: skip
        o = _block_opts(opts, rngs)
        self.attn, self.mlp = _block_parts(
            dim, num_heads, mlp_ratio, qkv_bias, drop, o, depth, rngs=rngs
        )
        self.norm1, self.norm2 = o["norm"](dim), o["norm"](dim)
        if init_values is not None:
            for n in (self.norm1, self.norm2):
                n.scale.set_value(jnp.full((dim,), init_values, jnp.float32))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.norm1(self.attn(x)))
        return x + self.drop_path(self.norm2(self.mlp(x)))


class _Branch(nnx.Module):
    """One ``ParallelThingsBlock`` branch: norm, attention or MLP, layer scale."""

    def __init__(self, norm, layer, ls, drop_path, kind, *, rngs):
        self.norm = norm
        setattr(self, kind, layer)
        self.kind, self.ls = kind, ls
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = getattr(self, self.kind)(self.norm(x))
        return self.drop_path(y if self.ls is None else self.ls(y))


class ParallelThingsBlock(nnx.Module):
    """Two attention branches summed, then two MLP branches summed."""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0,
        init_values=None, *, depth=0, num_parallel=2, rngs, **opts,
    ):  # fmt: skip
        o = _block_opts(opts, rngs)
        attns, ffns = [], []
        for _ in range(num_parallel):
            attn, mlp = _block_parts(dim, num_heads, mlp_ratio, qkv_bias, drop, o, depth, rngs=rngs)
            for branch, kind, layer in ((attns, "attn", attn), (ffns, "mlp", mlp)):
                ls = LayerScale(dim, init_values) if init_values else None
                branch.append(_Branch(o["norm"](dim), layer, ls, drop_path, kind, rngs=rngs))
        self.attns, self.ffns = nnx.List(attns), nnx.List(ffns)

    def __call__(self, x):
        x = x + sum(branch(x) for branch in self.attns)
        return x + sum(branch(x) for branch in self.ffns)


class ParallelScalingBlock(nnx.Module):
    """ViT-22B style block: one fused input projection feeds attention and the MLP in parallel.

    ``diff=True`` gives timm's ``DiffParallelScalingBlock`` (differential attention, fused
    output projection).
    """

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0,
        init_values=None, *, depth=0, diff=False, rngs, **opts,
    ):  # fmt: skip
        o = _block_opts(opts, rngs)
        self.num_heads, self.diff = num_heads, diff
        self.head_dim = dim // num_heads // (2 if diff else 1)
        hidden = int(mlp_ratio * dim)
        self.in_split = (hidden, hidden + dim, hidden + 2 * dim)
        self.in_norm = o["norm"](dim)
        self.in_proj = nnx.Linear(dim, hidden + 3 * dim, use_bias=qkv_bias, rngs=rngs)
        self.mlp_bias = None if qkv_bias else nnx.Param(jnp.zeros((hidden,)))
        self.q_norm = o["norm"](self.head_dim) if o["qk_norm"] else None
        self.k_norm = o["norm"](self.head_dim) if o["qk_norm"] else None
        self.act = o["act"]
        if diff:
            _init_diff(self, self.head_dim, depth, rngs=rngs)
            self.out_proj = nnx.Linear(dim + hidden, dim, use_bias=o["proj_bias"], rngs=rngs)
        else:
            self.attn_out_proj = nnx.Linear(dim, dim, use_bias=o["proj_bias"], rngs=rngs)
            self.mlp_out_proj = nnx.Linear(hidden, dim, use_bias=o["proj_bias"], rngs=rngs)
        self.ls = LayerScale(dim, init_values) if init_values is not None else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        x_mlp, q, k, v = jnp.split(self.in_proj(self.in_norm(x)), self.in_split, axis=-1)
        if self.mlp_bias is not None:
            x_mlp = x_mlp + self.mlp_bias[...]
        heads = 2 * self.num_heads if self.diff else self.num_heads
        q = q.reshape(B, N, heads, self.head_dim)
        k = k.reshape(B, N, heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        x_mlp = self.act(x_mlp)
        if self.diff:
            v = v.reshape(B, N, self.num_heads, -1)
            x_attn = _diff_attend(self, q, k, v)
            y = self.out_proj(jnp.concatenate([x_attn, x_mlp], axis=-1))
        else:
            v = v.reshape(B, N, heads, self.head_dim)
            x_attn = dot_product_attention(q, k, v).reshape(B, N, C)
            y = self.attn_out_proj(x_attn) + self.mlp_out_proj(x_mlp)
        return x + self.drop_path(y if self.ls is None else self.ls(y))


_BLOCKS = {
    "Block": Block,
    "ResPostBlock": ResPostBlock,
    "ParallelThingsBlock": ParallelThingsBlock,
    "ParallelScalingBlock": ParallelScalingBlock,
    "DiffParallelScalingBlock": lambda *a, **kw: ParallelScalingBlock(*a, diff=True, **kw),
}


class AttentionPoolLatent(nnx.Module):
    """timm ``AttentionPoolLatent`` (MAP head): one learned query attends over the tokens."""

    def __init__(self, dim, num_heads, mlp_ratio, norm, act, *, rngs):
        self.num_heads = num_heads
        init = nnx.initializers.truncated_normal(dim**-0.5)
        self.latent = nnx.Param(init(rngs.params(), (1, 1, dim)))
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.kv = nnx.Linear(dim, dim * 2, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.norm = norm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(jnp.broadcast_to(self.latent[...], (B, 1, C))).reshape(B, 1, h, C // h)
        kv = self.kv(x).reshape(B, N, 2, h, C // h)
        x = self.proj(dot_product_attention(q, kv[:, :, 0], kv[:, :, 1]).reshape(B, 1, C))
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class PatchEmbed(nnx.Module):
    """Non-overlapping patch projection, optionally followed by a norm (``embed_norm_layer``)."""

    def __init__(self, img_size, patch_size, in_chans, dim, bias=True, norm=None, *, rngs):
        self.img_size, self.patch_size = img_size, patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        p = (patch_size, patch_size)
        self.proj = nnx.Conv(in_chans, dim, p, strides=p, padding="VALID", use_bias=bias, rngs=rngs)
        self.norm = norm(dim) if norm is not None else None

    def __call__(self, x):
        x = self.proj(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x if self.norm is None else self.norm(x)


_DEFAULTS = dict(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
    qkv_bias=True, qk_norm=False, scale_attn_norm=False, scale_mlp_norm=False, proj_bias=True,
    init_values=None, class_token=True, pos_embed="learn", no_embed_class=False, reg_tokens=0,
    pre_norm=False, final_norm=True, fc_norm=None, pool_include_prefix=False,
    dynamic_img_size=False, embed_norm_layer=None, embed_norm_eps=1e-6, norm_layer="layernorm",
    norm_eps=1e-6, act_layer="gelu", block_fn="Block", mlp_layer="mlp", attn_layer="",
    distilled=False,
)  # fmt: skip


class VisionTransformer(ClassifierMixin, nnx.Module):
    """timm ``VisionTransformer``; keyword arguments follow timm's constructor (``_DEFAULTS``)."""

    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self, num_classes=1000, in_chans=3, global_pool="token", drop_rate=0.0,
        drop_path_rate=0.0, *, rngs, **overrides,
    ):  # fmt: skip
        cfg = {**_DEFAULTS, **overrides}
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        self.num_classes, self.global_pool = num_classes, global_pool
        dim, depth, heads = cfg["embed_dim"], cfg["depth"], cfg["num_heads"]
        self.num_features = self.embed_dim = dim
        act = _ACTS[cfg["act_layer"]]

        def norm(d):
            return _norm(cfg["norm_layer"], d, cfg["norm_eps"], rngs=rngs)

        embed_norm = None
        if cfg["embed_norm_layer"]:

            def embed_norm(d):
                return _norm(cfg["embed_norm_layer"], d, cfg["embed_norm_eps"], rngs=rngs)

        self.distilled = cfg["distilled"]
        self.no_embed_class = cfg["no_embed_class"]
        self.pool_include_prefix = cfg["pool_include_prefix"]
        self.patch_embed = PatchEmbed(
            cfg["img_size"], cfg["patch_size"], in_chans, dim, not cfg["pre_norm"], embed_norm,
            rngs=rngs,
        )  # fmt: skip
        init = nnx.initializers.normal(1e-6)
        n_cls = int(cfg["class_token"]) + int(self.distilled)
        self.num_prefix_tokens = n_cls + cfg["reg_tokens"]
        self.cls_token = nnx.Param(init(rngs.params(), (1, 1, dim))) if cfg["class_token"] else None
        self.dist_token = nnx.Param(init(rngs.params(), (1, 1, dim))) if self.distilled else None
        reg = cfg["reg_tokens"]
        self.reg_token = nnx.Param(init(rngs.params(), (1, reg, dim))) if reg else None
        n = self.patch_embed.num_patches
        embed_len = n if self.no_embed_class else n + self.num_prefix_tokens
        if cfg["pos_embed"] in ("", "none"):
            self.pos_embed = None
        else:
            pos_init = nnx.initializers.truncated_normal(0.02)
            self.pos_embed = nnx.Param(pos_init(rngs.params(), (1, embed_len, dim)))
        self.pos_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.norm_pre = norm(dim) if cfg["pre_norm"] else None
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        opts = {
            k: cfg[k]
            for k in (
                "qk_norm", "scale_attn_norm", "scale_mlp_norm", "proj_bias", "mlp_layer",
                "attn_layer", "norm_layer", "norm_eps",
            )
        }  # fmt: skip
        block_fn = _BLOCKS[cfg["block_fn"]]
        self.blocks = nnx.List(
            [
                block_fn(
                    dim,
                    heads,
                    cfg["mlp_ratio"],
                    cfg["qkv_bias"],
                    drop_rate,
                    dpr[i],
                    cfg["init_values"],
                    depth=i,
                    act=act,
                    rngs=rngs,
                    **opts,
                )  # fmt: skip
                for i in range(depth)
            ]
        )
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max") if cfg["fc_norm"] is None else cfg["fc_norm"]
        )
        final = cfg["final_norm"]
        self.norm = norm(dim) if final and not use_fc_norm else None
        self.attn_pool = (
            AttentionPoolLatent(dim, heads, cfg["mlp_ratio"], norm, act, rngs=rngs)
            if global_pool == "map"
            else None
        )
        self.fc_norm = norm(dim) if final and use_fc_norm else None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(dim, num_classes, rngs=rngs) if num_classes > 0 else None
        self.head_dist = (
            nnx.Linear(dim, num_classes, rngs=rngs) if self.distilled and num_classes > 0 else None
        )
        self.distilled_training = False
        self.deterministic = False

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            if (global_pool == "map") != (self.attn_pool is not None):
                raise ValueError("cannot add or remove attention pooling in reset_classifier()")
            self.global_pool = global_pool
        self.num_classes = num_classes
        rngs = nnx.Rngs(0)
        self.head = nnx.Linear(self.embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None
        if self.distilled:
            self.head_dist = (
                nnx.Linear(self.embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None
            )
        if self.deterministic:
            self.eval()

    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def _pos_embed(self, x):
        B = x.shape[0]
        prefix = [
            jnp.broadcast_to(t[...], (B, t.shape[1], self.embed_dim))
            for t in (self.cls_token, self.dist_token, self.reg_token)
            if t is not None
        ]
        if self.pos_embed is None:
            return jnp.concatenate(prefix + [x], axis=1)
        if self.no_embed_class:
            x = jnp.concatenate(prefix + [x + self.pos_embed[...]], axis=1)
        else:
            x = jnp.concatenate(prefix + [x], axis=1) + self.pos_embed[...]
        return self.pos_drop(x)

    def forward_intermediates(self, x, out_indices=None):
        """Patch-token feature maps (B, H, W, C) after each block."""
        from ..features import _select_features

        B = x.shape[0]
        gh, gw = self.patch_embed.grid_size
        x = self._pos_embed(self.patch_embed(x))
        if self.norm_pre is not None:
            x = self.norm_pre(x)
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x[:, self.num_prefix_tokens :].reshape(B, gh, gw, -1))
        return _select_features(feats, out_indices)

    def forward_features(self, x):
        x = self._pos_embed(self.patch_embed(x))
        if self.norm_pre is not None:
            x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        return x if self.norm is None else self.norm(x)

    def pool(self, x, pool_type=None):
        if self.attn_pool is not None:
            return self.attn_pool(x if self.pool_include_prefix else x[:, self.num_prefix_tokens :])
        pool_type = self.global_pool if pool_type is None else pool_type
        if pool_type == "token":
            return x[:, 0]
        if not pool_type:
            return x
        x = x if self.pool_include_prefix else x[:, self.num_prefix_tokens :]
        if pool_type == "avgmax":
            return 0.5 * (jnp.max(x, axis=1) + jnp.mean(x, axis=1))
        return jnp.max(x, axis=1) if pool_type == "max" else jnp.mean(x, axis=1)

    def forward_head(self, x, pre_logits=False):
        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            if pre_logits or self.head is None:
                return (x + x_dist) / 2
            x, x_dist = self.head(x), self.head_dist(x_dist)
            if self.distilled_training and not self.deterministic:
                return x, x_dist
            return (x + x_dist) / 2
        x = self.pool(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head_drop(x)
        if pre_logits or self.head is None:
            return x
        return self.head(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _make(name):
    overrides, ev = VIT_CFGS[name]
    overrides = dict(overrides)
    pool = overrides.pop("global_pool", "token")
    size = overrides["img_size"]

    def entry(**kwargs):
        kwargs.setdefault("global_pool", pool)
        model = VisionTransformer(**{**overrides, **kwargs})
        model.default_cfg = _cfg(**{"input_size": (3, size, size), **ev})
        return model

    entry.__name__ = name
    return entry


def register_all(names, module=__name__):
    """Registers ``names``; ``module`` sets the timm module they are listed under."""
    for name in names:
        entry = _make(name)
        entry.__module__ = module
        register_model(entry)


register_all(n for n in VIT_CFGS if not n.startswith("deit"))
