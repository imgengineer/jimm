"""Sequencer2D in flax nnx, NHWC. Mirrors timm.models.sequencer.

Token mixing runs bidirectional LSTMs (NNX LSTM cells) along every column and every row of the
feature map; their concatenated outputs are projected back to the channel
width. Blocks are pre-norm with an MLP, a 7x7 patch embedding starts the
network, and strided convolutions change the width between stages.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model


def _uniform(scale):
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(
        key, shape, dtype, -scale, scale
    )


def _bilstm(input_size, hidden_size, *, rngs):
    """Bidirectional single-layer LSTM from NNX cells, with PyTorch's gate order (i, f, g, o).

    PyTorch adds separate input and recurrent biases; giving ``dense_i`` a bias keeps
    timm's parameters. ``LSTM2d`` executes the cells via a fused input projection and
    batched ``jax.lax.scan``.
    """
    init = _uniform(1.0 / math.sqrt(hidden_size))

    def rnn():
        cell = nnx.OptimizedLSTMCell(
            input_size,
            hidden_size,
            kernel_init=init,
            recurrent_kernel_init=init,
            bias_init=init,
            rngs=rngs,
        )
        cell.dense_i = nnx.Linear(
            input_size, 4 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs
        )
        return nnx.RNN(cell, unroll=4, rngs=rngs)

    return nnx.Bidirectional(rnn(), rnn(), rngs=rngs)


class LSTM2d(nnx.Module):
    def __init__(self, dim, hidden_size, *, rngs):
        self.rnn_v = _bilstm(dim, hidden_size, rngs=rngs)
        self.rnn_h = _bilstm(dim, hidden_size, rngs=rngs)
        self.fc = nnx.Linear(
            4 * hidden_size, dim, kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs
        )

    def __call__(self, x):
        B, H, W, _C = x.shape
        cells = (
            self.rnn_v.forward_rnn.cell,
            self.rnn_v.backward_rnn.cell,
            self.rnn_h.forward_rnn.cell,
            self.rnn_h.backward_rnn.cell,
        )
        dtype = cells[0].dense_i.dtype or jnp.result_type(x, cells[0].dense_i.kernel[...])
        x_in = x.astype(dtype)
        # All four LSTM directions share the same input tensor: fuse their input projections
        # and both gate biases into one GEMM outside the recurrent scan.
        w_i = jnp.concatenate([c.dense_i.kernel[...].astype(dtype) for c in cells], axis=-1)
        b_all = jnp.concatenate(
            [(c.dense_i.bias[...] + c.dense_h.bias[...]).astype(dtype) for c in cells], axis=-1
        )
        gates = jnp.dot(x_in, w_i) + b_all
        g4 = gates.shape[-1] // 4
        hidden = g4 // 4
        gv_f, gv_b, gh_f, gh_b = jnp.split(gates, 4, axis=-1)
        w_h = jnp.stack([c.dense_h.kernel[...].astype(dtype) for c in cells], axis=0)

        def run_scan(seq, wh):
            # seq: (T, dirs, N, 4 * hidden), wh: (dirs, hidden, 4 * hidden)
            zero = jnp.zeros((seq.shape[1], seq.shape[2], hidden), dtype=dtype)

            def step(carry, x_t):
                c_prev, h_prev = carry
                h_proj = jax.lax.dot_general(h_prev, wh, (((2,), (1,)), ((0,), (0,))))
                i, f, g, o = jnp.split(x_t + h_proj, 4, axis=-1)
                c_new = jax.nn.sigmoid(f) * c_prev + jax.nn.sigmoid(i) * jnp.tanh(g)
                h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)
                return (c_new, h_new), h_new

            _, ys = jax.lax.scan(step, (zero, zero), seq, unroll=4)
            return ys

        vf = gv_f.transpose(1, 0, 2, 3).reshape(H, B * W, g4)
        vb = gv_b[:, ::-1].transpose(1, 0, 2, 3).reshape(H, B * W, g4)
        hf = gh_f.transpose(2, 0, 1, 3).reshape(W, B * H, g4)
        hb = gh_b[:, :, ::-1].transpose(2, 0, 1, 3).reshape(W, B * H, g4)
        if H == W:
            # Square maps share sequence length and batch count across all four directions.
            ys = run_scan(jnp.stack([vf, vb, hf, hb], axis=1), w_h)
            v_f, v_b, h_f, h_b = ys[:, 0], ys[::-1, 1], ys[:, 2], ys[::-1, 3]
        else:
            yv = run_scan(jnp.stack([vf, vb], axis=1), w_h[:2])
            yh = run_scan(jnp.stack([hf, hb], axis=1), w_h[2:])
            v_f, v_b, h_f, h_b = yv[:, 0], yv[::-1, 1], yh[:, 0], yh[::-1, 1]
        v_f = v_f.reshape(H, B, W, hidden).transpose(1, 0, 2, 3)
        v_b = v_b.reshape(H, B, W, hidden).transpose(1, 0, 2, 3)
        h_f = h_f.reshape(W, B, H, hidden).transpose(1, 2, 0, 3)
        h_b = h_b.reshape(W, B, H, hidden).transpose(1, 2, 0, 3)
        return self.fc(jnp.concatenate([v_f, v_b, h_f, h_b], axis=-1))


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, drop=0.0, *, rngs):
        init = dict(
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(1e-6),
            rngs=rngs,
        )
        self.fc1 = nnx.Linear(dim, hidden, **init)
        self.fc2 = nnx.Linear(hidden, dim, **init)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        return self.drop(self.fc2(self.drop(gelu(self.fc1(x)))))


class Sequencer2dBlock(nnx.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, drop=0.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.rnn_tokens = LSTM2d(dim, hidden_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp_channels = Mlp(dim, int(mlp_ratio * dim), drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        return x + self.drop_path(self.mlp_channels(self.norm2(x)))


class Sequencer2dStage(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        patch_size,
        hidden_size,
        mlp_ratio,
        downsample,
        drop,
        drop_path,
        *,
        rngs,
    ):
        self.downsample = (
            nnx.Conv(
                dim,
                dim_out,
                (patch_size, patch_size),
                strides=(patch_size, patch_size),
                padding="VALID",
                rngs=rngs,
            )
            if downsample
            else None
        )
        self.blocks = nnx.List(
            [
                Sequencer2dBlock(dim_out, hidden_size, mlp_ratio, drop, drop_path, rngs=rngs)
                for _ in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class Sequencer2d(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        layers=(4, 3, 8, 3),
        patch_sizes=(7, 2, 1, 1),
        embed_dims=(192, 384, 384, 384),
        hidden_sizes=(48, 96, 96, 96),
        mlp_ratios=(3.0, 3.0, 3.0, 3.0),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        p = patch_sizes[0]
        self.stem = nnx.Conv(
            in_chans, embed_dims[0], (p, p), strides=(p, p), padding="VALID", rngs=rngs
        )
        stages, prev = [], embed_dims[0]
        for i, dim in enumerate(embed_dims):
            # timm applies the same drop path rate to every block.
            stages.append(
                Sequencer2dStage(
                    prev,
                    dim,
                    layers[i],
                    patch_sizes[i],
                    hidden_sizes[i],
                    mlp_ratios[i],
                    downsample=i > 0,
                    drop=drop_rate,
                    drop_path=drop_path_rate,
                    rngs=rngs,
                )
            )
            prev = dim
        self.stages = nnx.List(stages)
        self.num_features = embed_dims[-1]
        self.norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._linear(num_classes, rngs)

    def _linear(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        zeros = nnx.initializers.zeros
        return nnx.Linear(self.num_features, num_classes, kernel_init=zeros, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.head = self._linear(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "sequencer2d_s": (4, 3, 8, 3),
    "sequencer2d_m": (4, 3, 14, 3),
    "sequencer2d_l": (8, 8, 16, 4),
}


def _make(name):
    layers = _CFGS[name]

    def entry(**kwargs):
        model = Sequencer2d(layers, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
