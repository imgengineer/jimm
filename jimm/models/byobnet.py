"""ByobNet ("bring your own blocks") in flax nnx, NHWC. Mirrors timm.models.byobnet and byoanet.

One network assembled from per-block configs recorded from timm's builder
(``_byobnet_cfgs``): basic, bottleneck, dark, edge, RepVGG, MobileOne and
self-attention blocks with conv/avg-pool shortcuts; squeeze-excite, ECA,
global-context and bilinear-attention-transform channel attention;
bottleneck (BoTNet), halo (HaloNet) and lambda (LambdaNet) self-attention;
BatchNorm or EvoNorm-S0a; optional average-pool anti-aliasing; and plain,
MLP or CLIP attention-pool heads. Convolutions use PyTorch symmetric padding.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model
from ._byobnet_cfgs import BLOCK_DEFAULTS, BYOB_CFGS

_ACTS = {"relu": nnx.relu, "silu": nnx.silu, "gelu": lambda x: jax.nn.gelu(x, approximate=False)}


def _num_groups(group_size, chs):
    return chs // group_size if group_size else 1


def _conv(in_chs, out_chs, kernel, stride=1, dilation=1, groups=1, bias=False, *, rngs):
    p = ((stride - 1) + dilation * (kernel - 1)) // 2
    return nnx.Conv(
        in_chs, out_chs, (kernel, kernel), strides=stride, padding=((p, p), (p, p)),
        kernel_dilation=dilation, feature_group_count=groups, use_bias=bias, rngs=rngs,
    )  # fmt: skip


def _avg2(x, stride=2, ceil=False):
    """AvgPool2d(2, stride[, ceil_mode=True, count_include_pad=False])."""
    B, H, W, C = x.shape
    window, strides = (1, 2, 2, 1), (1, stride, stride, 1)
    if not ceil:
        return jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, "VALID") / 4.0
    oh, ow = -(-(H - 2) // stride) + 1, -(-(W - 2) // stride) + 1
    pad = (
        (0, 0),
        (0, max((oh - 1) * stride + 2 - H, 0)),
        (0, max((ow - 1) * stride + 2 - W, 0)),
        (0, 0),
    )
    total = jax.lax.reduce_window(jnp.pad(x, pad), 0.0, jax.lax.add, window, strides, "VALID")
    ones = jnp.pad(jnp.ones((1, H, W, 1), x.dtype), pad)
    return total / jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, "VALID")


class EvoNorm2dS0a(nnx.Module):
    """``x * sigmoid(v * x) / group_std(x)`` (std of the input), then an affine transform."""

    def __init__(self, chs, group_size=16, apply_act=True, eps=1e-3):
        self.scale = nnx.Param(jnp.ones(chs))
        self.bias = nnx.Param(jnp.zeros(chs))
        self.v = nnx.Param(jnp.ones(chs)) if apply_act else None
        self.groups, self.eps = chs // group_size, eps

    def __call__(self, x):
        B, H, W, C = x.shape
        g = x.astype(jnp.promote_types(x.dtype, jnp.float32)).reshape(B, H, W, self.groups, -1)
        std = jnp.sqrt(g.var(axis=(1, 2, 4), keepdims=True) + self.eps)
        std = jnp.broadcast_to(std, g.shape).reshape(B, H, W, C).astype(x.dtype)
        if self.v is not None:
            x = x * jax.nn.sigmoid(x * self.v[...])
        return (x / std) * self.scale[...] + self.bias[...]


class NormAct(nnx.Module):
    """BatchNorm (+ activation) or EvoNorm-S0a (activation built in)."""

    def __init__(self, chs, layers, apply_act=True, *, rngs):
        norm = layers["norm"]
        if isinstance(norm, (list, tuple)):
            self.evo = True
            self.norm = EvoNorm2dS0a(chs, norm[1].get("group_size", 16), apply_act)
        else:
            self.evo = False
            self.norm = BatchNorm(chs, epsilon=1e-5, rngs=rngs)
        self.act = layers["act"] if apply_act else None

    def __call__(self, x):
        x = self.norm(x)
        return _ACTS[self.act](x) if (self.act and not self.evo) else x


class ConvNormAct(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride=1, dilation=1, groups=1, apply_act=True,
                 layers=None, aa=True, *, rngs):  # fmt: skip
        self.use_aa = aa and layers["aa"] == "avg" and stride > 1
        s = 1 if self.use_aa else stride
        self.conv = _conv(in_chs, out_chs, kernel, s, dilation, groups, rngs=rngs)
        na = NormAct(out_chs, layers, apply_act, rngs=rngs)
        self.bn, self.evo, self.act = na.norm, na.evo, na.act

    def __call__(self, x):
        x = self.bn(self.conv(x))
        if self.act and not self.evo:
            x = _ACTS[self.act](x)
        return _avg2(x) if self.use_aa else x


# ----------------------------------------------------------------------------- channel attention


class SEModule(nnx.Module):
    def __init__(self, chs, rd_ratio=1 / 16, rd_divisor=8, *, rngs):
        rd = make_divisible(chs * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nnx.Conv(chs, rd, (1, 1), rngs=rngs)
        self.fc2 = nnx.Conv(rd, chs, (1, 1), rngs=rngs)

    def __call__(self, x):
        s = self.fc2(nnx.relu(self.fc1(x.mean(axis=(1, 2), keepdims=True))))
        return x * jax.nn.sigmoid(s)


class EcaModule(nnx.Module):
    def __init__(self, chs, gamma=2, beta=1, *, rngs):
        t = int(abs(math.log(chs, 2) + beta) / gamma)
        k = max(t if t % 2 else t + 1, 3)
        self.conv = nnx.Conv(1, 1, (k,), padding=(((k - 1) // 2,) * 2,), use_bias=False, rngs=rngs)

    def __call__(self, x):
        s = self.conv(x.mean(axis=(1, 2))[:, :, None])[:, :, 0]
        return x * jax.nn.sigmoid(s)[:, None, None, :]


class GlobalContext(nnx.Module):
    """Global context attention with additive fusion (timm 'gca')."""

    def __init__(self, chs, rd_ratio=1 / 8, *, rngs):
        rd = make_divisible(chs * rd_ratio, 1, round_limit=0.0)
        self.conv_attn = nnx.Conv(chs, 1, (1, 1), rngs=rngs)
        self.mlp_add = _ConvMlpLN(chs, rd, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        attn = jax.nn.softmax(self.conv_attn(x).reshape(B, H * W), axis=-1)
        context = jnp.einsum("bnc,bn->bc", x.reshape(B, H * W, C), attn)[:, None, None, :]
        return x + self.mlp_add(context)


class _ConvMlpLN(nnx.Module):
    def __init__(self, chs, hidden, *, rngs):
        self.fc1 = nnx.Conv(chs, hidden, (1, 1), rngs=rngs)
        self.norm = nnx.LayerNorm(hidden, epsilon=1e-6, rngs=rngs)
        self.fc2 = nnx.Conv(hidden, chs, (1, 1), rngs=rngs)

    def __call__(self, x):
        return self.fc2(nnx.relu(self.norm(self.fc1(x))))


class _CNA(nnx.Module):
    """ConvNormAct with BatchNorm + ReLU (timm's defaults inside BAT)."""

    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv = _conv(in_chs, out_chs, 1, rngs=rngs)
        self.bn = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.bn(self.conv(x)))


class BilinearAttnTransform(nnx.Module):
    def __init__(self, chs, block_size, groups, *, rngs):
        self.block_size, self.groups = block_size, groups
        self.conv1 = _CNA(chs, groups, rngs=rngs)
        n = block_size * block_size * groups
        self.conv_p = nnx.Conv(groups, n, (block_size, 1), padding="VALID", rngs=rngs)
        self.conv_q = nnx.Conv(groups, n, (1, block_size), padding="VALID", rngs=rngs)
        self.conv2 = _CNA(chs, chs, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        bs, g = self.block_size, self.groups
        out = self.conv1(x)
        # adaptive max pool to (bs, 1) and (1, bs); sizes divide evenly.
        rp = out.reshape(B, bs, H // bs, W, g).max(axis=(2, 3))[:, :, None, :]
        cp = out.reshape(B, H, bs, W // bs, g).max(axis=(1, 3))[:, None, :, :]
        p = jax.nn.sigmoid(self.conv_p(rp).reshape(B, g, bs, bs))
        q = jax.nn.sigmoid(self.conv_q(cp).reshape(B, g, bs, bs))
        p = p / p.sum(axis=3, keepdims=True)
        q = q / q.sum(axis=2, keepdims=True)
        p = jnp.repeat(p, C // g, axis=1)  # (B, C, bs, bs)
        q = jnp.repeat(q, C // g, axis=1)
        p = jnp.kron(p, jnp.eye(H // bs, dtype=p.dtype)) if H // bs > 1 else p
        q = jnp.kron(q, jnp.eye(W // bs, dtype=q.dtype)) if W // bs > 1 else q
        y = jnp.einsum("bchk,bkwc->bhwc", p, x)
        y = jnp.einsum("bhkc,bckw->bhwc", y, q)
        return self.conv2(y)


class BatNonLocalAttn(nnx.Module):
    def __init__(self, chs, block_size=7, groups=2, rd_ratio=0.25, rd_divisor=8, *, rngs):
        rd = make_divisible(chs * rd_ratio, rd_divisor)
        self.conv1 = _CNA(chs, rd, rngs=rngs)
        self.ba = BilinearAttnTransform(rd, block_size, groups, rngs=rngs)
        self.conv2 = _CNA(rd, chs, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, broadcast_dims=(1, 2), rngs=rngs)

    def __call__(self, x):
        return self.dropout(self.conv2(self.ba(self.conv1(x)))) + x


_ATTN = {"SEModule": SEModule, "EcaModule": EcaModule, "BatNonLocalAttn": BatNonLocalAttn}


def _make_attn(desc, chs, rngs):
    if desc is None:
        return None
    name, kw = desc
    if name == "GlobalContext":
        return GlobalContext(chs, rngs=rngs)
    return _ATTN[name](chs, **kw, rngs=rngs)


# ----------------------------------------------------------------------------- self attention


def _rel_logits_1d(q, rel_k):
    """q (..., H, W, d), rel_k (2*win-1, d) -> (..., H, W, win) with index u - w + win - 1."""
    win = (rel_k.shape[0] + 1) // 2
    W = q.shape[-2]
    idx = np.arange(win)[None, :] - np.arange(W)[:, None] + win - 1  # (W, win)
    return jnp.einsum("...wd,wud->...wu", q, rel_k[idx])


def _rel_pos_bias(q, height_rel, width_rel, h, w):
    """q (N, h*w, d) -> (N, h*w, win_h*win_w) relative logits (timm PosEmbedRel)."""
    qq = q.reshape(-1, h, w, q.shape[-1])
    lw = _rel_logits_1d(qq, width_rel)  # (N, h, w, win_w)  [i, j, u]
    lh = _rel_logits_1d(jnp.swapaxes(qq, 1, 2), height_rel)  # (N, w, h, win_h) [j, i, a]
    win_h, win_w = lh.shape[-1], lw.shape[-1]
    bias = jnp.swapaxes(lh, 1, 2)[..., :, None] + lw[..., None, :]  # (N, h, w, a, u)
    return bias.reshape(q.shape[0], h * w, win_h * win_w)


class BottleneckAttn(nnx.Module):
    def __init__(self, dim, feat_size, stride=1, num_heads=4, dim_head=None, qk_ratio=1.0, *, rngs):
        self.num_heads = num_heads
        self.dqk = dim_head or make_divisible(dim * qk_ratio, divisor=8) // num_heads
        self.dv = dim // num_heads
        self.qkv = nnx.Conv(
            dim, num_heads * (2 * self.dqk + self.dv), (1, 1), use_bias=False, rngs=rngs
        )
        self.pos_embed = _PosEmbedRel(feat_size[0], feat_size[1], self.dqk, rngs=rngs)
        self.stride = stride

    def __call__(self, x):
        B, H, W, C = x.shape
        h, dqk, dv = self.num_heads, self.dqk, self.dv
        qkv = self.qkv(x).reshape(B, H * W, -1)
        q = qkv[..., : h * dqk].reshape(B, H * W, h, dqk)
        k = qkv[..., h * dqk : 2 * h * dqk].reshape(B, H * W, h, dqk)
        v = qkv[..., 2 * h * dqk :].reshape(B, H * W, h, dv)
        qh = jnp.transpose(q, (0, 2, 1, 3)).reshape(B * h, H * W, dqk)
        bias = self.pos_embed(qh, H, W).reshape(B, h, H * W, H * W)
        out = dot_product_attention(q, k, v, bias=bias).reshape(B, H, W, h * dv)
        return _avg2(out) if self.stride == 2 else out


class _PosEmbedRel(nnx.Module):
    def __init__(self, win_h, win_w, dim_head, *, rngs):
        init = nnx.initializers.normal(dim_head**-0.5)
        self.height_rel = nnx.Param(init(rngs.params(), (2 * win_h - 1, dim_head)))
        self.width_rel = nnx.Param(init(rngs.params(), (2 * win_w - 1, dim_head)))

    def __call__(self, q, h, w):
        return _rel_pos_bias(q, self.height_rel[...], self.width_rel[...], h, w)


class HaloAttn(nnx.Module):
    def __init__(self, dim, stride=1, num_heads=8, dim_head=None, block_size=8, halo_size=3,
                 qk_ratio=1.0, avg_down=False, *, rngs):  # fmt: skip
        self.num_heads = num_heads
        self.dqk = dim_head or make_divisible(dim * qk_ratio, divisor=8) // num_heads
        self.dv = dim // num_heads
        self.block_size, self.halo = block_size, halo_size
        self.win = block_size + 2 * halo_size
        self.block_stride, use_pool = 1, False
        if stride > 1:
            use_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_pool else stride
        self.bs_ds = block_size // self.block_stride
        self.use_pool = use_pool
        self.q = nnx.Conv(dim, num_heads * self.dqk, (1, 1), strides=self.block_stride,
                          padding="VALID", use_bias=False, rngs=rngs)  # fmt: skip
        self.kv = nnx.Conv(dim, num_heads * (self.dqk + self.dv), (1, 1), use_bias=False, rngs=rngs)
        self.pos_embed = _PosEmbedRel(self.win, self.win, self.dqk, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        h, dqk, dv, bs, bsd, win = (
            self.num_heads,
            self.dqk,
            self.dv,
            self.block_size,
            self.bs_ds,
            self.win,
        )
        nh, nw = H // bs, W // bs
        q = self.q(x).reshape(B, nh, bsd, nw, bsd, h, dqk)
        q = jnp.transpose(q, (0, 5, 1, 3, 2, 4, 6)).reshape(B * h, nh * nw, bsd * bsd, dqk)
        kv = jnp.pad(self.kv(x), ((0, 0), (self.halo,) * 2, (self.halo,) * 2, (0, 0)))
        # Overlapping win x win windows at stride block_size (unfold).
        rows = np.arange(nh)[:, None] * bs + np.arange(win)[None, :]  # (nh, win)
        cols = np.arange(nw)[:, None] * bs + np.arange(win)[None, :]
        kv = kv[:, rows[:, None, :, None], cols[None, :, None, :]]  # (B, nh, nw, win, win, C')
        kv = kv.reshape(B, nh * nw, win * win, h, dqk + dv)
        kv = jnp.transpose(kv, (0, 3, 1, 2, 4)).reshape(B * h, nh * nw, win * win, dqk + dv)
        k, v = kv[..., :dqk], kv[..., dqk:]
        bias = self.pos_embed(q.reshape(-1, bsd * bsd, dqk), bsd, bsd).reshape(
            B * h, nh * nw, bsd * bsd, win * win
        )
        logits = jnp.einsum("bnqd,bnkd->bnqk", q, k) * dqk**-0.5 + bias
        out = jnp.einsum("bnqk,bnkd->bnqd", jax.nn.softmax(logits, axis=-1), v)  # (Bh, nb, q, dv)
        out = out.reshape(B, h, nh, nw, bsd, bsd, dv)
        out = jnp.transpose(out, (0, 2, 4, 3, 5, 1, 6)).reshape(B, nh * bsd, nw * bsd, h * dv)
        return _avg2(out) if self.use_pool else out


class LambdaLayer(nnx.Module):
    def __init__(
        self, dim, feat_size, stride=1, num_heads=4, dim_head=16, r=9, qk_ratio=1.0, *, rngs
    ):
        self.num_heads = num_heads
        self.dqk = dim_head or make_divisible(dim * qk_ratio, divisor=8) // num_heads
        self.dv = dim // num_heads
        self.qkv = nnx.Conv(
            dim, num_heads * self.dqk + self.dqk + self.dv, (1, 1), use_bias=False, rngs=rngs
        )
        self.norm_q = BatchNorm(num_heads * self.dqk, epsilon=1e-5, rngs=rngs)
        self.norm_v = BatchNorm(self.dv, epsilon=1e-5, rngs=rngs)
        self.stride = stride
        if r is not None:
            self.conv_lambda = nnx.Conv(
                1, self.dqk, (r, r), padding=((r // 2,) * 2,) * 2, rngs=rngs
            )
            self.pos_emb = None
        else:
            fh, fw = feat_size
            self.conv_lambda = None
            self.pos_emb = nnx.Param(
                nnx.initializers.truncated_normal(0.02)(
                    rngs.params(), (2 * fh - 1, 2 * fw - 1, self.dqk)
                )
            )
            pos = np.stack(np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")).reshape(2, -1)
            rel = pos[:, None, :] - pos[:, :, None]
            self.rel_idx = nnx.Variable(jnp.asarray(np.stack([rel[0] + fh - 1, rel[1] + fw - 1])))

    def __call__(self, x):
        B, H, W, C = x.shape
        h, dqk, dv, M = self.num_heads, self.dqk, self.dv, H * W
        qkv = self.qkv(x)
        q = self.norm_q(qkv[..., : h * dqk]).reshape(B, M, h, dqk)
        k = qkv[..., h * dqk : h * dqk + dqk].reshape(B, M, dqk)
        v = self.norm_v(qkv[..., h * dqk + dqk :]).reshape(B, M, dv)
        k = jax.nn.softmax(k, axis=1)
        content = jnp.einsum("bmhk,bkv->bmhv", q, jnp.einsum("bmk,bmv->bkv", k, v))
        if self.pos_emb is None:
            vv = jnp.transpose(v.reshape(B, H, W, dv), (0, 3, 1, 2)).reshape(B * dv, H, W, 1)
            lam = self.conv_lambda(vv).reshape(B, dv, M, dqk)  # [b, c, m, k]
            position = jnp.einsum("bmhk,bcmk->bmhc", q, lam)
        else:
            idx = self.rel_idx[...]
            pe = self.pos_emb[...][idx[0], idx[1]]  # (M, M, dqk)
            lam = jnp.einsum("mnk,bnv->bmkv", pe, v)
            position = jnp.einsum("bmhk,bmkv->bmhv", q, lam)
        out = (content + position).reshape(B, H, W, h * dv)
        return _avg2(out) if self.stride == 2 else out


def _make_self_attn(desc, chs, stride, feat_size, rngs):
    name, kw = desc
    if name == "BottleneckAttn":
        return BottleneckAttn(chs, feat_size, stride, **kw, rngs=rngs)
    if name == "HaloAttn":
        return HaloAttn(chs, stride, **kw, rngs=rngs)
    return LambdaLayer(chs, feat_size, stride, **kw, rngs=rngs)


# ----------------------------------------------------------------------------- blocks


class Shortcut(nnx.Module):
    def __init__(self, kind, in_chs, out_chs, stride, layers, *, rngs):
        self.pool_stride = stride if kind == "avg" and stride > 1 else 0
        if kind == "avg":
            self.conv = ConvNormAct(in_chs, out_chs, 1, apply_act=False, layers=layers, rngs=rngs)
        else:
            self.conv = ConvNormAct(in_chs, out_chs, 1, stride, apply_act=False, layers=layers,
                                    aa=True, rngs=rngs)  # fmt: skip

    def __call__(self, x):
        if self.pool_stride:
            x = _avg2(x, self.pool_stride, ceil=True)
        return self.conv(x)


def _shortcut(b, layers, rngs):
    if b["in_chs"] != b["out_chs"] or b["stride"] != 1:
        if not b["downsample"]:
            return None
        return Shortcut(b["downsample"], b["in_chs"], b["out_chs"], b["stride"], layers, rngs=rngs)
    return "identity"


class ByoBlock(nnx.Module):
    """basic / bottle / dark / edge / self_attn residual blocks (timm field names)."""

    def __init__(self, b, layers, drop_path, *, rngs):
        kind, cin, cout, s = b["type"], b["in_chs"], b["out_chs"], b["stride"]
        dil = b["dilation"]
        k = b.get("kernel_size", 3)
        br = b["bottle_ratio"]
        cna = dict(layers=layers, rngs=rngs)
        sc = _shortcut(b, layers, rngs)
        self.has_shortcut = sc is not None
        self.shortcut = sc if isinstance(sc, Shortcut) else None
        self.kind = kind
        attn = attn_last = conv2b = conv2 = None
        if kind in ("bottle", "self_attn"):
            mid = make_divisible((cin if b.get("bottle_in") else cout) * br)
            g = _num_groups(b["group_size"], mid)
            self.conv1_1x1 = ConvNormAct(cin, mid, 1, **cna)
            if kind == "bottle":
                self.conv2_kxk = ConvNormAct(mid, mid, k, s, dil[0], g, aa=True, **cna)
                if b.get("extra_conv"):
                    conv2b = ConvNormAct(mid, mid, k, 1, dil[1], g, **cna)
                attn = _make_attn(b["attn"], mid, rngs)
            else:
                if b.get("extra_conv"):
                    conv2 = ConvNormAct(mid, mid, k, s, dil[0], g, aa=True, **cna)
                    s = 1
                self.conv2_kxk = conv2
                self.self_attn = _make_self_attn(b["self_attn"], mid, s, b.get("feat_size"), rngs)
                self.post_attn = NormAct(mid, layers, rngs=rngs)
            self.conv3_1x1 = ConvNormAct(mid, cout, 1, apply_act=False, **cna)
        elif kind == "basic":
            mid = make_divisible(cout * br)
            g = _num_groups(b["group_size"], mid)
            self.conv1_kxk = ConvNormAct(cin, mid, k, s, dil[0], aa=True, **cna)
            self.conv2_kxk = ConvNormAct(mid, cout, k, 1, dil[1], g, apply_act=False, **cna)
            attn_last = _make_attn(b["attn"], cout, rngs)
        elif kind == "dark":
            mid = make_divisible(cout * br)
            g = _num_groups(b["group_size"], mid)
            self.conv1_1x1 = ConvNormAct(cin, mid, 1, **cna)
            self.conv2_kxk = ConvNormAct(
                mid, cout, k, s, dil[0], g, apply_act=False, aa=True, **cna
            )
            attn_last = _make_attn(b["attn"], cout, rngs)
        else:  # edge
            mid = make_divisible(cout * br)
            g = _num_groups(b["group_size"], mid)
            self.conv1_kxk = ConvNormAct(cin, mid, k, s, dil[0], g, aa=True, **cna)
            attn = _make_attn(b["attn"], mid, rngs)
            self.conv2_1x1 = ConvNormAct(mid, cout, 1, apply_act=False, **cna)
        self.attn, self.attn_last, self.conv2b_kxk = attn, attn_last, conv2b
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.act = None if b.get("linear_out") else layers["act"]

    def __call__(self, x):
        shortcut = x
        kind = self.kind
        if kind == "bottle":
            x = self.conv2_kxk(self.conv1_1x1(x))
            if self.conv2b_kxk is not None:
                x = self.conv2b_kxk(x)
            if self.attn is not None:
                x = self.attn(x)
            x = self.conv3_1x1(x)
        elif kind == "self_attn":
            x = self.conv1_1x1(x)
            if self.conv2_kxk is not None:
                x = self.conv2_kxk(x)
            x = self.conv3_1x1(self.post_attn(self.self_attn(x)))
        elif kind == "basic":
            x = self.conv2_kxk(self.conv1_kxk(x))
        elif kind == "dark":
            x = self.conv2_kxk(self.conv1_1x1(x))
        else:
            x = self.conv1_kxk(x)
            if self.attn is not None:
                x = self.attn(x)
            x = self.conv2_1x1(x)
        if self.attn_last is not None:
            x = self.attn_last(x)
        x = self.drop_path(x)
        if self.has_shortcut:
            x = x + (shortcut if self.shortcut is None else self.shortcut(shortcut))
        return _ACTS[self.act](x) if self.act else x


class RepBlock(nnx.Module):
    """RepVGG (kxk + 1x1 + identity BN) or MobileOne (k parallel kxk + 1x1 scale + identity)."""

    def __init__(self, b, layers, drop_path, *, rngs):
        cin, cout, s = b["in_chs"], b["out_chs"], b["stride"]
        g = _num_groups(b["group_size"], cin)
        k = b.get("kernel_size", 3)
        self.one = b["type"] == "one"
        use_ident = cin == cout and s == 1
        self.identity = (
            NormAct(cout, layers, apply_act=False, rngs=rngs).norm if use_ident else None
        )
        if self.one:
            n = b.get("num_conv_branches", 1)
            self.conv_kxk = nnx.List(
                [
                    ConvNormAct(
                        cin, cout, k, s, groups=g, apply_act=False, layers=layers, rngs=rngs
                    )
                    for _ in range(n)
                ]  # fmt: skip
            )
            self.conv_scale = (
                ConvNormAct(cin, cout, 1, s, groups=g, apply_act=False, layers=layers, rngs=rngs)
                if k > 1
                else None
            )
        else:
            self.conv_kxk = ConvNormAct(
                cin, cout, k, s, groups=g, apply_act=False, layers=layers, rngs=rngs
            )
            self.conv_1x1 = ConvNormAct(
                cin, cout, 1, s, groups=g, apply_act=False, layers=layers, rngs=rngs
            )
        self.drop_path = DropPath(drop_path if use_ident else 0.0, rngs=rngs)
        self.attn = _make_attn(b["attn"], cout, rngs)
        self.act = layers["act"]

    def __call__(self, x):
        if self.one:
            out = self.conv_scale(x) if self.conv_scale is not None else 0.0
            for ck in self.conv_kxk:
                out = out + ck(x)
        else:
            out = self.conv_1x1(x) + self.conv_kxk(x)
        if self.identity is not None:
            out = self.drop_path(out) + self.identity(x)
        if self.attn is not None:
            out = self.attn(out)
        return _ACTS[self.act](out)


def _block(b, layers, dpr, rngs):
    full = {**BLOCK_DEFAULTS, **b}
    if full["type"] in ("rep", "one"):
        return RepBlock(full, layers, dpr, rngs=rngs)
    return ByoBlock(full, layers, dpr, rngs=rngs)


class Stem(nnx.Module):
    """timm byobnet Stem: a chain of strided/plain conv(-norm-act) layers and an optional pool."""

    def __init__(self, in_chs, out_chs, kernel=3, pool="maxpool", num_rep=3, num_act=None,
                 chs_decay=0.5, layers=None, *, rngs):  # fmt: skip
        if isinstance(out_chs, (list, tuple)):
            num_rep, chs = len(out_chs), list(out_chs)
        else:
            chs = [round(out_chs * chs_decay**i) for i in range(num_rep)][::-1]
        strides = [2] + [1] * (num_rep - 1)
        if not pool:
            strides[-1] = 2
        num_act = num_rep if num_act is None else num_act
        acts = [False] * (num_rep - num_act) + [True] * num_act
        prev = in_chs
        for i, (c, s, na) in enumerate(zip(chs, strides, acts)):
            layer = (
                ConvNormAct(prev, c, kernel, s, layers=layers, rngs=rngs)
                if na
                else _conv(prev, c, kernel, s, rngs=rngs)
            )
            setattr(self, f"conv{i + 1}", layer)
            prev = c
        self.num_rep, self.pool, self.out_chs = num_rep, pool, prev

    def __call__(self, x):
        for i in range(self.num_rep):
            x = getattr(self, f"conv{i + 1}")(x)
        pool = (self.pool or "").lower()
        if pool == "max2":
            return jax.lax.reduce_window(
                x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID"
            )
        if pool == "avg2":
            return _avg2(x)
        if "max" in pool:
            return nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        if "avg" in pool:
            pad = ((0, 0), (1, 1), (1, 1), (0, 0))
            total = jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 3, 3, 1), (1, 2, 2, 1), pad)
            ones = jnp.ones((1, *x.shape[1:3], 1), x.dtype)
            cnt = jax.lax.reduce_window(ones, 0.0, jax.lax.add, (1, 3, 3, 1), (1, 2, 2, 1), pad)
            return total / cnt
        return x


def _stem(in_chs, cfg, layers, rngs):
    out, t, pool = cfg["out_chs"], cfg["stem_type"], cfg["pool_type"]
    if "quad" in t:
        return Stem(
            in_chs,
            out,
            num_rep=4,
            num_act=2 if "quad2" in t else None,
            pool=pool,
            layers=layers,
            rngs=rngs,
        )
    if "tiered" in t:
        return Stem(in_chs, (3 * out // 8, out // 2, out), pool=pool, layers=layers, rngs=rngs)
    if "deep" in t:
        return Stem(in_chs, out, num_rep=3, chs_decay=1.0, pool=pool, layers=layers, rngs=rngs)
    if t in ("rep", "one"):
        b = dict(
            type=t, in_chs=in_chs, out_chs=out, stride=2, group_size=None, attn=cfg.get("attn")
        )
        return RepBlock(b, layers, 0.0, rngs=rngs)
    if "7x7" in t:
        if pool:
            return Stem(in_chs, out, 7, num_rep=1, pool=pool, layers=layers, rngs=rngs)
        return ConvNormAct(in_chs, out, 7, 2, layers=layers, rngs=rngs)
    if isinstance(out, (list, tuple)):
        return Stem(in_chs, out, 3, pool=pool, layers=layers, rngs=rngs)
    if pool:
        return Stem(in_chs, out, 3, num_rep=1, pool=pool, layers=layers, rngs=rngs)
    return ConvNormAct(in_chs, out, 3, 2, layers=layers, rngs=rngs)


class AttentionPool2d(nnx.Module):
    """CLIP attention pooling: mean token query over learned absolute position embeddings."""

    def __init__(self, dim, feat, head_dim=64, *, rngs):
        self.num_heads = dim // head_dim
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.pos_embed = nnx.Param(
            nnx.initializers.normal(dim**-0.5)(rngs.params(), (feat * feat + 1, dim))
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = jnp.concatenate([x.mean(axis=1, keepdims=True), x], axis=1) + self.pos_embed[...]
        h = self.num_heads
        shape = (B, H * W + 1, h, C // h)
        out = dot_product_attention(
            self.q(x).reshape(shape), self.k(x).reshape(shape), self.v(x).reshape(shape)
        )
        return out.reshape(B, H * W + 1, C)


class ByobNet(ClassifierMixin, nnx.Module):
    _classifier_attr = "fc"

    def __init__(self, model, stages, num_classes=1000, in_chans=3, global_pool=None,
                 drop_rate=0.0, drop_path_rate=0.0, img_size=224, *, rngs):  # fmt: skip
        layers = dict(act=model["act"], norm=model["norm"], aa=model["aa"])
        stem_layers = dict(layers, aa="")
        self.stem = _stem(
            in_chans, {**model["stem"], "attn": model.get("stem_attn")}, stem_layers, rngs
        )
        total = sum(n for st in stages for _, n in st)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        blocks_all, k = [], 0
        for st in stages:
            blocks = []
            for b, n in st:
                for j in range(n):
                    blocks.append(_block(b, layers, rates[k], rngs))
                    k += 1
            blocks_all.append(nnx.List(blocks))
        self.stages = nnx.List(blocks_all)
        prev = {**BLOCK_DEFAULTS, **stages[-1][-1][0]}["out_chs"]
        nf = model["num_features"]
        if nf:
            self.num_features = int(round(model["width_factor"] * nf))
            self.final_conv = ConvNormAct(prev, self.num_features, 1, layers=layers, rngs=rngs)
        else:
            self.num_features, self.final_conv = prev, None
        self.head_type = model["head_type"]
        self.global_pool = global_pool if global_pool is not None else model["global_pool"]
        self.num_classes = num_classes
        head_norm = pre_logits = attn_pool = None
        self.head_act = model["act"]
        if self.head_type == "mlp":
            head_norm = BatchNorm(self.num_features, epsilon=1e-5, rngs=rngs)
            pre_logits = nnx.Linear(self.num_features, model["head_hidden_size"], rngs=rngs)
            self.num_features = model["head_hidden_size"]
        elif self.head_type == "attn_abs":
            attn_pool = AttentionPool2d(self.num_features, img_size // 32, rngs=rngs)
        self.head_norm, self.pre_logits, self.attn_pool = head_norm, pre_logits, attn_pool
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        return nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return self.final_conv(x) if self.final_conv is not None else x

    def forward_head(self, x):
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.head_drop(x)
            if self.fc is not None:
                x = self.fc(x)
            return x[:, 0] if self.global_pool == "token" else x[:, 1:]
        if self.global_pool == "avg":
            x = x.mean(axis=(1, 2))
        elif self.global_pool == "max":
            x = x.max(axis=(1, 2))
        if self.pre_logits is not None:
            x = _ACTS[self.head_act](self.pre_logits(self.head_norm(x)))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _make(name):
    model, stages, ev = BYOB_CFGS[name]

    def entry(**kwargs):
        size = ev["input_size"][1]
        img = model["img_size"] or size
        kwargs.setdefault("img_size", img[0] if isinstance(img, (tuple, list)) else img)
        net = ByobNet(model, stages, **kwargs)
        net.default_cfg = _cfg(**ev)
        return net

    entry.__name__ = name
    return entry


def register_all(names):
    for name in names:
        register_model(_make(name))


_BYOANET = {
    "botnet26t_256", "botnet50ts_256", "eca_botnext26ts_256", "eca_halonext26ts",
    "halo2botnet50ts_256", "halonet26t", "halonet50ts", "halonet_h1", "haloregnetz_b",
    "lambda_resnet26rpt_256", "lambda_resnet26t", "lambda_resnet50ts", "lamhalobotnet50ts_256",
    "sebotnet33ts_256", "sehalonet33ts",
}  # fmt: skip
register_all([n for n in BYOB_CFGS if n not in _BYOANET])
