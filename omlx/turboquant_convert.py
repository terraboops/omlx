# SPDX-License-Identifier: Apache-2.0
"""Streaming TurboQuant 3.5-bit conversion.

Converts a model's weights from fp16 to TQ3.5 (rotate + 3-bit quantize) without
loading the full model into memory. Processes tensors one-by-one from safetensors,
applying orthogonal rotation before quantization.

Memory usage: ~3-4GB regardless of model size (same as oq.py streaming path).

The output model can be loaded normally with mlx-lm; TurboQuantLinear layers
are used instead of QuantizedLinear when the rotation matrices are present.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_MAX_SHARD_BYTES = 4 * 1024 * 1024 * 1024  # 4GB per shard


def _wht(x: mx.array) -> mx.array:
    """Walsh-Hadamard Transform via butterfly — same as turboquant_kv._wht."""
    import math
    shape = x.shape
    D = shape[-1]
    flat = x.reshape(-1, D).astype(mx.float32)
    h = 1
    while h < D:
        flat_r = flat.reshape(-1, D // (2 * h), 2, h)
        a = flat_r[:, :, 0, :]
        b = flat_r[:, :, 1, :]
        flat_r = mx.stack([a + b, a - b], axis=2)
        flat = flat_r.reshape(-1, D)
        h *= 2
    flat = flat / math.sqrt(D)
    return flat.reshape(shape)


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _rotation_matrix(dim: int, seed: int = 0) -> mx.array:
    """Build dense WHT rotation matrix: block-diagonal diag(signs) @ H.

    For power-of-2 dimensions: standard WHT.
    For non-power-of-2: block-diagonal WHT where each block is the largest
    power of 2 that fits. E.g. dim=768 → 512 + 256 blocks.
    Preserves orthogonality for all dimensions.
    """
    if dim > 0 and (dim & (dim - 1)) == 0:
        # Power of 2 — standard WHT
        key = mx.random.key(seed)
        uniform = mx.random.uniform(shape=(dim,), key=key)
        signs = mx.where(uniform > 0.5, mx.ones(dim), -mx.ones(dim)).astype(mx.float32)
        mx.eval(signs)
        I = mx.eye(dim, dtype=mx.float32)
        H = _wht(I)
        R = signs[:, None] * H
        mx.eval(R)
        return R

    # Non-power-of-2: decompose into power-of-2 blocks
    import numpy as np
    R_np = np.zeros((dim, dim), dtype=np.float32)
    remaining = dim
    offset = 0
    block_seed = seed

    while remaining > 0:
        # Find largest power of 2 ≤ remaining
        block_size = 1
        while block_size * 2 <= remaining:
            block_size *= 2

        # Build WHT block
        key = mx.random.key(block_seed)
        uniform = mx.random.uniform(shape=(block_size,), key=key)
        signs = mx.where(uniform > 0.5, mx.ones(block_size), -mx.ones(block_size)).astype(mx.float32)
        mx.eval(signs)
        I = mx.eye(block_size, dtype=mx.float32)
        H = _wht(I)
        block = np.array(signs[:, None] * H)

        R_np[offset:offset+block_size, offset:offset+block_size] = block
        offset += block_size
        remaining -= block_size
        block_seed += 1

    R = mx.array(R_np, dtype=mx.float32)
    mx.eval(R)
    return R


def _get_layer_index(tensor_name: str) -> int:
    """Extract layer index from tensor name like 'model.layers.5.self_attn.q_proj.weight'."""
    parts = tensor_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def _is_linear_weight(tensor_name: str, shape: tuple) -> bool:
    """Check if a tensor is a linear layer weight that should be rotated+quantized."""
    if not tensor_name.endswith(".weight"):
        return False
    if len(shape) < 2:
        return False
    # Skip: embeddings, norms, lm_head, and SSM-specific recurrent params.
    # Mamba's in_proj/out_proj are standard linear layers — safe to quantize.
    # Only the SSM recurrent params (A, D, dt, conv1d) must stay fp16.
    lower = tensor_name.lower()
    skip = [
        "embed", "norm", "lm_head", "wte", "wpe", "rotary", "rope",
        "conv1d",       # Mamba convolutional state (recurrent, sensitive)
        "a_log",        # SSM state transition (recurrent, exponential)
        "d_param",      # SSM skip connection
        "dt_proj",      # SSM time-step projection (recurrent)
        "dt_bias",      # SSM time-step bias
    ]
    return not any(s in lower for s in skip)


def convert_turbo35_streaming(
    model_path: str,
    output_path: str,
    group_size: int = 64,
    bits: int = 3,
    fp16_layers: int = 0,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> None:
    """Stream-convert model weights to TQ3.5 (rotate + 3-bit quantize).

    Never loads the full model. Processes tensors one-by-one.
    Memory: ~3-4GB regardless of model size.

    Args:
        model_path: Source model directory (fp16 safetensors).
        output_path: Output directory for TQ3.5 model.
        group_size: Quantization group size.
        bits: Quantization bits (3 for TQ3.5).
        fp16_layers: First N layers kept in fp16.
        progress_callback: Optional fn(phase, pct).
    """
    source = Path(model_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Output path already exists: {output_path}")

    output.mkdir(parents=True, exist_ok=True)
    cb = progress_callback or (lambda phase, pct: None)

    # Copy config and tokenizer files
    cb("loading", 5.0)
    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Copy non-weight files
    for fname in source.iterdir():
        if fname.suffix not in (".safetensors",) and fname.name != ".gitattributes":
            dest = output / fname.name
            if not dest.exists():
                shutil.copy2(fname, dest)

    cb("loading", 10.0)

    # Find safetensor files
    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No .safetensors files found in {model_path}")

    # Load all tensor names and shapes (metadata only, not the data)
    all_weights = {}
    for sf_path in weight_files:
        shard = mx.load(str(sf_path), return_metadata=False)
        all_weights.update(shard)
        del shard

    tensor_names = list(all_weights.keys())
    total = len(tensor_names)
    logger.info(f"TQ3.5 streaming: {total} tensors in {len(weight_files)} shards")

    cb("quantizing", 15.0)

    # Process tensors
    out_shard_data = {}
    out_shard_idx = 0
    weight_map = {}
    converted = 0
    kept_fp16 = 0
    per_layer_config = {}

    for i, name in enumerate(tensor_names):
        w = all_weights.pop(name)
        shape = w.shape

        layer_idx = _get_layer_index(name)

        if _is_linear_weight(name, shape) and not (0 <= layer_idx < fp16_layers):
            in_dim = shape[-1]

            # Can we quantize this dimension?
            if in_dim % group_size != 0:
                out_shard_data[name] = w.astype(mx.bfloat16)
                del w
            else:
                # Shared rotation matrix per dimension (data-oblivious, only depends on dim)
                R = _rotation_matrix(in_dim, seed=in_dim)
                base = name[:-7] if name.endswith(".weight") else name

                # Handle 2D and 3D weights
                if len(shape) == 3:
                    # 3D MoE expert weight: (num_experts, out_dim, in_dim)
                    # Rotate each expert: W[e] @ R for all e
                    w_f32 = w.astype(mx.float32)
                    w_rotated = w_f32 @ mx.broadcast_to(R, (shape[0], *R.shape))
                    del w_f32

                    # Split input_linear → gate_proj + up_proj (granitemoehybrid sanitize)
                    # and quantize each sub-tensor to QuantizedSwitchLinear format
                    if "input_linear" in name:
                        expert_hidden = shape[1]
                        gate = w_rotated[:, :expert_hidden // 2, :]
                        up = w_rotated[:, expert_hidden // 2:, :]

                        for sub_name, sub_w in [("gate_proj", gate), ("up_proj", up)]:
                            qw, sc, *rest = mx.quantize(sub_w, group_size=group_size, bits=bits)
                            bi = rest[0] if rest else None
                            mx.eval(qw, sc)
                            if bi is not None:
                                mx.eval(bi)
                            switch_base = base.replace("input_linear", f"switch_mlp.{sub_name}")
                            out_shard_data[f"{switch_base}.weight"] = qw
                            out_shard_data[f"{switch_base}.scales"] = sc
                            if bi is not None:
                                out_shard_data[f"{switch_base}.biases"] = bi
                            per_layer_config[switch_base] = {"bits": bits, "group_size": group_size, "mode": "affine"}

                    elif "output_linear" in name:
                        qw, sc, *rest = mx.quantize(w_rotated, group_size=group_size, bits=bits)
                        bi = rest[0] if rest else None
                        mx.eval(qw, sc)
                        if bi is not None:
                            mx.eval(bi)
                        switch_base = base.replace("output_linear", "switch_mlp.down_proj")
                        out_shard_data[f"{switch_base}.weight"] = qw
                        out_shard_data[f"{switch_base}.scales"] = sc
                        if bi is not None:
                            out_shard_data[f"{switch_base}.biases"] = bi
                        per_layer_config[switch_base] = {"bits": bits, "group_size": group_size, "mode": "affine"}

                    else:
                        # Other 3D: quantize under original name
                        qw, sc, *rest = mx.quantize(w_rotated, group_size=group_size, bits=bits)
                        bi = rest[0] if rest else None
                        mx.eval(qw, sc)
                        if bi is not None:
                            mx.eval(bi)
                        out_shard_data[f"{base}.weight"] = qw
                        out_shard_data[f"{base}.scales"] = sc
                        if bi is not None:
                            out_shard_data[f"{base}.biases"] = bi
                        per_layer_config[base] = {"bits": bits, "group_size": group_size, "mode": "affine"}

                    del w, w_rotated, R
                    converted += 1
                    # Flush eagerly after large 3D tensors
                    mx.synchronize()
                    mx.clear_cache()
                    continue
                else:
                    w_rotated = w.astype(mx.float32) @ R

                qw, scales, *rest = mx.quantize(w_rotated, group_size=group_size, bits=bits)
                biases = rest[0] if rest else None
                mx.eval(qw, scales)
                if biases is not None:
                    mx.eval(biases)

                out_shard_data[f"{base}.weight"] = qw
                out_shard_data[f"{base}.scales"] = scales
                if biases is not None:
                    out_shard_data[f"{base}.biases"] = biases

                per_layer_config[base] = {
                    "bits": bits,
                    "group_size": group_size,
                    "mode": "affine",
                }
                converted += 1
                del w, w_rotated, qw, scales, R
        else:
            # Keep as-is (fp16 layers, embeddings, norms, etc.)
            if w.dtype == mx.float32:
                w = w.astype(mx.bfloat16)
            out_shard_data[name] = w
            if _is_linear_weight(name, shape) and 0 <= layer_idx < fp16_layers:
                kept_fp16 += 1
            del w

        # Flush shard when large enough
        current_bytes = sum(v.nbytes for v in out_shard_data.values())
        if current_bytes >= _MAX_SHARD_BYTES:
            shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
            shard_path = output / shard_name
            mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
            for k in out_shard_data:
                weight_map[k] = shard_name
            out_shard_idx += 1
            out_shard_data = {}
            mx.synchronize()
            mx.clear_cache()

        pct = 15.0 + 80.0 * (i + 1) / total
        cb("quantizing", pct)

    # Flush remaining
    if out_shard_data:
        shard_name = f"model-{out_shard_idx + 1:05d}-of-PLACEHOLDER.safetensors"
        shard_path = output / shard_name
        mx.save_safetensors(str(shard_path), out_shard_data, metadata={"format": "mlx"})
        for k in out_shard_data:
            weight_map[k] = shard_name
        out_shard_idx += 1

    # Fix shard count in filenames
    total_shards = out_shard_idx
    final_weight_map = {}
    for old_name in sorted(set(weight_map.values())):
        new_name = old_name.replace("PLACEHOLDER", f"{total_shards:05d}")
        old_path = output / old_name
        new_path = output / new_name
        if old_path.exists():
            old_path.rename(new_path)
        for k, v in weight_map.items():
            if v == old_name:
                final_weight_map[k] = new_name

    # Write weight map index
    index = {
        "metadata": {"total_size": sum(
            (output / fn).stat().st_size for fn in set(final_weight_map.values())
        )},
        "weight_map": final_weight_map,
    }
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Update config with quantization info
    config["quantization"] = {
        "group_size": group_size,
        "bits": bits,
        "mode": "affine",
        "turboquant_rotation": True,
    }
    if per_layer_config:
        config["quantization"]["per_layer"] = per_layer_config

    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    cb("saving", 100.0)

    logger.info(
        f"TQ3.5 conversion complete: {converted} layers rotated+quantized to {bits}-bit, "
        f"{kept_fp16} layers kept fp16. Output: {output}"
    )
