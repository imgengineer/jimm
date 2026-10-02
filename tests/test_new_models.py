"""Architecture and lifecycle regressions for the five timm 1.0.30 model families."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import jimm
from jimm.models._conv import ConvTranspose
from jimm.models._rope_vit import apply_rope, axial_rope, resample_pos_embed_grid
from jimm.models.deepseek_vit import DeepseekVitAligner, RmsNormFp32
from jimm.models.qwen3_vit import Qwen3VitPatchMerger
from jimm.train import cross_entropy, make_cached_eval_step, make_cached_train_step

# Counts from timm 1.0.30 with num_classes=5, including native projector defaults.
PARAM_COUNTS = {
    "lowformer_b0": 12507477,
    "lowformer_b1": 16343317,
    "lowformer_b15": 31433833,
    "lowformer_b2": 42438717,
    "lowformer_b3": 55493925,
    "lowformer_e1": 16348073,
    "lowformer_e2": 21157669,
    "lowformer_e3": 39723813,
    "iformer_t": 2630741,
    "iformer_s": 6243973,
    "iformer_m": 8524349,
    "iformer_l": 14396733,
    "iformer_l2": 24008549,
    "iformer_h": 98831477,
    "iformer_m_distilled": 8527042,
    "iformer_l_distilled": 14399426,
    "iformer_l2_distilled": 24012138,
    "efficientvim_m1": 5720278,
    "efficientvim_m2": 12498902,
    "efficientvim_m3": 15040034,
    "efficientvim_m4": 18042365,
    "efficientvim_m1_dist": 5725102,
    "efficientvim_m2_dist": 12505966,
    "efficientvim_m3_dist": 15047898,
    "efficientvim_m4_dist": 18050229,
    "qwen3_vit_88m": 87418373,
    "qwen3_vit_88m_merge": 100008197,
    "qwen3_vit_88m_enc": 100003072,
    "qwen3_vit_306m": 305461253,
    "qwen3_vit_306m_merge": 330640389,
    "qwen3_vit_306m_enc": 330630144,
    "qwen3_vit_416m": 415012469,
    "qwen3_vit_416m_merge": 459870965,
    "qwen3_vit_416m_enc": 459845360,
    "deepseek_vit_412m": 411847685,
    "deepseek_vit_412m_align": 485278725,
    "deepseek_vit_412m_enc": 485253120,
}

SMALL_CONFIGS = {
    "iformer": dict(dims=(8, 16, 24, 32), depths=(1, 1, 3, 3), attn_groups=(0, 0, 1, 1)),
    "efficientvim": dict(embed_dim=(16, 24, 32), depths=(1, 1, 1), state_dim=(4, 3, 2)),
    "lowformer": dict(
        width_list=(8, 16, 24, 32, 40), depth_list=(0, 1, 1, 1, 1), head_widths=(48, 56)
    ),
    "qwen3_vit": dict(
        img_size=32,
        patch_size=4,
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        pos_embed_grid_size=4,
    ),
    "deepseek_vit": dict(
        img_size=32, patch_size=4, embed_dim=16, depth=2, num_heads=2, mlp_ratio=2
    ),
}


@pytest.mark.parametrize("name", PARAM_COUNTS)
def test_registered_architectures_match_timm_parameter_counts(name):
    model = nnx.eval_shape(lambda: jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(0)))
    count = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)))
    assert count == PARAM_COUNTS[name]
    assert jimm.get_default_cfg(name)["input_size"] == model.default_cfg["input_size"]


@pytest.mark.parametrize(
    "name,family",
    [
        ("iformer_t", "iformer"),
        ("efficientvim_m1", "efficientvim"),
        ("lowformer_b0", "lowformer"),
        ("qwen3_vit_88m", "qwen3_vit"),
        ("deepseek_vit_412m", "deepseek_vit"),
    ],
)
def test_new_architecture_jit_training_features_and_classifier_reset(name, family):
    model = jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(0), **SMALL_CONFIGS[family])
    x = jax.random.normal(jax.random.key(1), (2, 32, 32, 3))
    labels = jnp.array([0, 1])
    model.train()

    def loss_fn(m, inputs, targets):
        return cross_entropy(m(inputs), targets)

    loss, grads = nnx.jit(nnx.value_and_grad(loss_fn))(model, x, labels)
    leaves = jax.tree.leaves(grads)
    assert bool(jnp.isfinite(loss))
    assert leaves and all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)

    model.eval()
    logits = nnx.jit(model)(x)
    assert logits.shape == (2, 5)
    np.testing.assert_allclose(
        logits, model.forward_head(model.forward_features(x)), atol=1e-5, rtol=1e-5
    )
    features = jimm.create_feature_extractor(model)(x)
    assert len(features) == (3 if family == "efficientvim" else 2 if "vit" in family else 4)
    assert all(feature.ndim == 4 and feature.shape[0] == 2 for feature in features)
    selected = nnx.jit(jimm.create_feature_extractor(model, out_indices=(-1, 0)))(x)
    np.testing.assert_allclose(selected[0], features[-1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(selected[1], features[0], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="out of range"):
        jimm.create_feature_extractor(model, out_indices=(len(features),))(x)

    pooled = model.forward_head(model.forward_features(x), pre_logits=True)
    model.reset_classifier(0)
    assert model.get_classifier() is None
    assert model(x).shape == (2, model.num_features)
    np.testing.assert_allclose(model(x), pooled, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("groups", [1, 4])
def test_grouped_transpose_convolution_preserves_bf16_compute_and_fp32_weights(groups):
    model = ConvTranspose(4, 4, 4, 2, 1, groups=groups, rngs=nnx.Rngs(0))
    images = jax.random.normal(jax.random.key(1), (2, 3, 4, 4))
    expected = model(images)

    @nnx.jit
    def forward(m, inputs):
        return m(inputs)

    actual = forward(model, images.astype(jnp.bfloat16))
    assert actual.shape == (2, 6, 8, 4) and actual.dtype == jnp.bfloat16
    np.testing.assert_allclose(actual.astype(jnp.float32), expected, rtol=0.03, atol=0.03)
    loss, grads = nnx.jit(
        nnx.value_and_grad(
            lambda m: jnp.square(m(images.astype(jnp.bfloat16)).astype(jnp.float32)).mean()
        )
    )(model)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(grads))
    assert model.kernel[...].dtype == model.bias[...].dtype == jnp.float32


@pytest.mark.parametrize(
    "name,family",
    [
        ("iformer_t", "iformer"),
        ("efficientvim_m1", "efficientvim"),
        ("lowformer_b0", "lowformer"),
        ("qwen3_vit_88m", "qwen3_vit"),
        ("deepseek_vit_412m", "deepseek_vit"),
    ],
)
def test_new_models_cached_amp_training_preserves_master_weights(name, family):
    model = jimm.create_model(name, num_classes=3, rngs=nnx.Rngs(0), **SMALL_CONFIGS[family])
    optimizer = jimm.make_optimizer(model, 1e-3, 0.01, 1, 1)
    images = jax.random.normal(jax.random.key(1), (2, 32, 32, 3))
    labels = jnp.array([0, 1])
    model.train()
    loss, _ = make_cached_train_step(model, optimizer, amp=True)(images, labels, 0.1)
    assert bool(jnp.isfinite(loss)) and int(optimizer.step[...]) == 1
    parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert all(leaf.dtype == jnp.float32 and bool(jnp.isfinite(leaf).all()) for leaf in parameters)
    assert all(
        leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(nnx.state(model, nnx.BatchStat))
    )
    model.eval()
    eval_loss, _ = make_cached_eval_step(model, amp=True)(images, labels)
    assert bool(jnp.isfinite(eval_loss))


@pytest.mark.parametrize(
    "name,family",
    [
        ("iformer_m_distilled", "iformer"),
        ("efficientvim_m1_dist", "efficientvim"),
    ],
)
def test_distilled_heads_follow_train_and_eval_modes(name, family):
    model = jimm.create_model(name, num_classes=3, rngs=nnx.Rngs(0), **SMALL_CONFIGS[family])
    x = jax.random.normal(jax.random.key(2), (2, 32, 32, 3))

    @nnx.jit
    def forward(m, inputs):
        return m(inputs)

    model.set_distilled_training(True)
    model.train()
    logits = forward(model, x)
    assert isinstance(logits, tuple) and len(logits) == 2
    assert all(logit.shape == (2, 3) for logit in logits)
    model.eval()
    logits = forward(model, x)
    assert logits.shape == (2, 3)
    model.reset_classifier(4)
    assert model(x).shape == (2, 4)
    model.reset_classifier(0)
    assert model(x).shape == (2, model.num_features)


@pytest.mark.parametrize(
    "name,family,native",
    [
        ("qwen3_vit_88m_merge", "qwen3_vit", False),
        ("qwen3_vit_88m_enc", "qwen3_vit", True),
        ("deepseek_vit_412m_align", "deepseek_vit", False),
        ("deepseek_vit_412m_enc", "deepseek_vit", True),
    ],
)
def test_vlm_projection_is_independent_of_class_count(name, family, native):
    model = jimm.create_model(
        name, num_classes=5, out_features=8, rngs=nnx.Rngs(0), **SMALL_CONFIGS[family]
    )
    model.eval()
    x = jnp.ones((2, 32, 48, 3))
    output = nnx.jit(model)(x)
    if native:
        assert model.num_classes == 0 and model.get_classifier() is None
        assert output.shape == (2, 24 if family == "qwen3_vit" else 12, 8)
        model.reset_classifier(0, global_pool="avg")
        assert model(x).shape == (2, 16)
        with pytest.raises(ValueError, match="Classifier"):
            model.reset_classifier(2)
    else:
        assert output.shape == (2, 5)
        assert model.num_features == 8
        model.reset_classifier(0)
        assert model(x).shape == (2, 8)


def test_deepseek_dynamic_padding_and_qwen_invalid_patch_grid():
    kwargs = dict(SMALL_CONFIGS["deepseek_vit"], out_features=8, dynamic_img_pad=True)
    model = jimm.create_model("deepseek_vit_412m_enc", rngs=nnx.Rngs(0), **kwargs)
    model.eval()
    x = jnp.ones((1, 17, 19, 3))
    assert model.forward_features(x).shape == (1, 5, 5, 16)
    assert nnx.jit(model)(x).shape == (1, 4, 8)
    strict = jimm.create_model(
        "qwen3_vit_88m_enc", out_features=8, rngs=nnx.Rngs(0), **SMALL_CONFIGS["qwen3_vit"]
    )
    strict.eval()
    with pytest.raises(ValueError, match="patch_size"):
        strict(x)
    with pytest.raises(ValueError, match="merge_size"):
        strict(jnp.ones((1, 20, 24, 3)))


@pytest.mark.parametrize(
    "name,family",
    [
        ("qwen3_vit_88m", "qwen3_vit"),
        ("deepseek_vit_412m", "deepseek_vit"),
    ],
)
def test_rope_attention_dropout_obeys_model_modes(name, family):
    model = jimm.create_model(
        name,
        num_classes=3,
        attn_drop_rate=0.5,
        proj_drop_rate=0.2,
        rngs=nnx.Rngs(0),
        **SMALL_CONFIGS[family],
    )
    x = jax.random.normal(jax.random.key(3), (2, 32, 32, 3))

    @nnx.jit
    def forward(m, inputs):
        return m(inputs)

    model.eval()
    np.testing.assert_array_equal(forward(model, x), forward(model, x))
    model.train()
    assert not np.allclose(forward(model, x), forward(model, x))


def test_projectors_preserve_their_distinct_spatial_channel_orders():
    x = jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 4, 2)
    qwen = Qwen3VitPatchMerger(2, merge_size=2, rngs=nnx.Rngs(0))
    deepseek = DeepseekVitAligner(2, downsample_ratio=2, out_features=4, rngs=nnx.Rngs(0))
    np.testing.assert_array_equal(
        qwen.merge(x), [[[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]]
    )
    np.testing.assert_array_equal(
        deepseek.unshuffle(x), [[[0, 2, 8, 10, 1, 3, 9, 11], [4, 6, 12, 14, 5, 7, 13, 15]]]
    )
    padded = deepseek.unshuffle(x[:, :, :3])
    np.testing.assert_array_equal(padded[:, -1], [[4, 0, 12, 0, 5, 0, 13, 0]])


def test_qwen_bilinear_position_interpolation_uses_aligned_corners():
    pos = jnp.array([[[0.0], [2.0], [4.0], [6.0]]])
    expected = np.array([[[0, 1, 2], [2, 3, 4], [4, 5, 6]]], dtype=np.float32).reshape(1, 9, 1)
    np.testing.assert_allclose(resample_pos_embed_grid(pos, 2, (3, 3)), expected)
    np.testing.assert_array_equal(resample_pos_embed_grid(pos, 2, (2, 2)), pos)
    np.testing.assert_allclose(resample_pos_embed_grid(pos, 2, (1, 3)), [[[0], [1], [2]]])


def test_axial_rope_half_rotation_and_fp32_rms_norm():
    rope = axial_rope(2, 3, 8)
    np.testing.assert_allclose(rope[0][0], 0)
    np.testing.assert_allclose(rope[1][0], 1)
    expected_phase = np.array([1, 0.01, 2, 0.02, 1, 0.01, 2, 0.02], dtype=np.float32)
    np.testing.assert_allclose(rope[0][-1], np.sin(expected_phase), atol=1e-6)
    x = jnp.arange(48, dtype=jnp.float32).reshape(1, 6, 1, 8)
    expected = np.asarray(x)[0, -1, 0] * np.cos(expected_phase)
    expected += np.array([-44, -45, -46, -47, 40, 41, 42, 43]) * np.sin(expected_phase)
    np.testing.assert_allclose(apply_rope(x, rope)[0, -1, 0], expected, atol=1e-5)
    norm = RmsNormFp32(4, rngs=nnx.Rngs(0))
    large = jnp.array([[300, 400, 500, 600]], dtype=jnp.float16)
    normalized = norm(large)
    assert normalized.dtype == jnp.float16
    assert bool(jnp.isfinite(normalized).all())
    assert bool(jnp.any(normalized != 0))
