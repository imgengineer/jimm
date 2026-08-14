"""MLP-Mixer and ResMLP in flax nnx. Mirrors timm.models.mlp_mixer."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed
from ..registry import register_model, _cfg


class MixerBlock(nnx.Module):
    def __init__(self, num_tokens, dim, tokens_mlp, channels_mlp, drop=0.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.token_mlp = Mlp(num_tokens, tokens_mlp, drop, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.channel_mlp = Mlp(dim, channels_mlp, drop, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        y = self.norm1(x).transpose(0, 2, 1)
        y = self.token_mlp(y).transpose(0, 2, 1)
        x = x + self.drop_path(y)
        return x + self.drop_path(self.channel_mlp(self.norm2(x)))


class MlpMixer(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, num_blocks=8, embed_dim=512,
                 mlp_ratio=(0.5, 4.0), num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        dpr = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nnx.List([
            MixerBlock(n, embed_dim, int(n * mlp_ratio[0]), int(embed_dim * mlp_ratio[1]),
                       drop_rate, dpr[i], rngs=rngs) for i in range(num_blocks)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.patch_embed(x).reshape(x.shape[0], -1, self.num_features)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = jnp.mean(x, axis=1) if self.global_pool == "avg" else x[:, 0]
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.head is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.head = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _mixer(patch, dim, blocks, **kwargs):
    model = MlpMixer(patch_size=patch, embed_dim=dim, num_blocks=blocks, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixer_b16_224(**kwargs):
    return _mixer(16, 768, 12, **kwargs)


@register_model
def mixer_b32_224(**kwargs):
    return _mixer(32, 768, 12, img_size=224, **kwargs)


@register_model
def mixer_l16_224(**kwargs):
    return _mixer(16, 1024, 24, **kwargs)


class ResMLPBlock(nnx.Module):
    def __init__(self, num_tokens, dim, mlp_ratio=4.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.token_fc = nnx.Linear(num_tokens, num_tokens, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.channel_mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)

    def __call__(self, x):
        y = self.norm1(x).transpose(0, 2, 1)
        x = x + self.token_fc(y).transpose(0, 2, 1)
        return x + self.channel_mlp(self.norm2(x))


class ResMLP(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, num_blocks=12, embed_dim=384,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.blocks = nnx.List([ResMLPBlock(n, embed_dim, rngs=rngs) for _ in range(num_blocks)])
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.patch_embed(x).reshape(x.shape[0], -1, self.num_features)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = jnp.mean(x, axis=1) if self.global_pool == "avg" else x[:, 0]
        return self.head(x) if self.head is not None else x

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.head is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.head = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


@register_model
def resmlp_12_224(**kwargs):
    model = ResMLP(num_blocks=12, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def resmlp_24_224(**kwargs):
    model = ResMLP(num_blocks=24, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def resmlp_36_224(**kwargs):
    model = ResMLP(num_blocks=36, **kwargs)
    model.default_cfg = _cfg()
    return model
