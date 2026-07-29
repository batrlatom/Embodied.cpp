#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a dense single-camera Qantara checkpoint to GGUF."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:  # pragma: no cover - user-facing startup error
    raise SystemExit(
        "failed to import gguf. Run patches/init_third_party.sh first. "
        f"Original error: {exc}"
    )


ARCH = "qantara"
KV = lambda name: f"{ARCH}.{name}"


def _tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(default if value is None else value)


def _validate_config(cfg: dict[str, Any]) -> None:
    backbone = cfg["backbone"]
    unsupported: list[str] = []
    if backbone.get("name") != "vit_tiny":
        unsupported.append("non-ViT-Tiny backbone")
    if _tuple(cfg.get("camera_names"), ("agentview",)) != ("agentview",):
        unsupported.append("multicamera")
    if _tuple(cfg.get("task_names"), ()) or cfg.get("task_action_adapter", False):
        unsupported.append("task conditioning")
    if cfg.get("goal_conditioning", False):
        unsupported.append("goal conditioning")
    if cfg.get("prefix_horizon", 0):
        unsupported.append("prefix dynamics")
    if cfg.get("robotttt", False):
        unsupported.append("RoboTTT")
    if unsupported:
        raise SystemExit(
            "minimal Qantara converter does not yet support: " + ", ".join(unsupported)
        )


def _add_metadata(
    writer: "gguf.GGUFWriter",
    cfg: dict[str, Any],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    flow_steps: int,
) -> None:
    backbone = cfg["backbone"]
    writer.add_string(KV("architecture"), ARCH)
    writer.add_string(KV("format_stage"), "full_policy_f32")
    writer.add_uint32(KV("format_version"), 1)
    writer.add_uint32(KV("latent_dim"), int(cfg["latent_dim"]))
    writer.add_uint32(KV("hidden_dim"), int(cfg["hidden_dim"]))
    writer.add_uint32(KV("depth"), int(cfg["depth"]))
    writer.add_uint32(KV("heads"), int(cfg["heads"]))
    writer.add_uint32(KV("head_dim"), int(cfg["head_dim"]))
    writer.add_uint32(KV("mlp_dim"), int(cfg["mlp_dim"]))
    writer.add_uint32(KV("num_frames"), int(cfg["num_frames"]))
    writer.add_uint32(KV("action_dim"), int(cfg["action_dim"]))
    writer.add_uint32(KV("frameskip"), int(cfg["frameskip"]))
    writer.add_uint32(KV("action_block_dim"), int(cfg["action_dim"] * cfg["frameskip"]))
    writer.add_uint32(KV("flow_steps"), flow_steps)
    writer.add_float32(KV("rms_norm_eps"), 1.0e-6)
    writer.add_float32(KV("rope_theta"), 10_000.0)
    writer.add_uint32(KV("image_size"), int(backbone["image_size"]))
    writer.add_uint32(KV("patch_size"), int(backbone["patch_size"]))
    writer.add_uint32(KV("vision_depth"), int(backbone["vit_depth"]))
    writer.add_uint32(KV("vision_heads"), int(backbone["vit_heads"]))
    writer.add_uint32(KV("vision_mlp_dim"), int(backbone["vit_mlp_dim"]))
    writer.add_array(KV("action_mean"), action_mean.float().tolist())
    writer.add_array(KV("action_std"), action_std.float().tolist())


def _add_tensor(
    writer: "gguf.GGUFWriter", name: str, tensor: torch.Tensor
) -> None:
    if tensor.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        return
    value = tensor.detach().float().contiguous().cpu().numpy()
    writer.add_tensor(
        name,
        value,
        raw_dtype=gguf.GGMLQuantizationType.F32,
    )


def _encoder_name(source: str) -> str:
    direct = {
        "encoder.backbone.cls": "qantara.vision.cls",
        "encoder.backbone.position": "qantara.vision.position",
        "encoder.backbone.patch.weight": "qantara.vision.patch.weight",
        "encoder.backbone.patch.bias": "qantara.vision.patch.bias",
        "encoder.backbone.blocks.norm.weight": "qantara.vision.norm.weight",
        "encoder.backbone.blocks.norm.bias": "qantara.vision.norm.bias",
        "encoder.projector.0.weight": "qantara.encoder.proj_in.weight",
        "encoder.projector.0.bias": "qantara.encoder.proj_in.bias",
        "encoder.projector.3.weight": "qantara.encoder.proj_out.weight",
        "encoder.projector.3.bias": "qantara.encoder.proj_out.bias",
    }
    if source in direct:
        return direct[source]
    match = re.fullmatch(
        r"encoder\.backbone\.blocks\.layers\.(\d+)\.(.+)", source
    )
    if not match:
        raise KeyError(f"unmapped encoder tensor {source}")
    layer, suffix = match.groups()
    suffixes = {
        "self_attn.in_proj_weight": "attn_qkv.weight",
        "self_attn.in_proj_bias": "attn_qkv.bias",
        "self_attn.out_proj.weight": "attn_out.weight",
        "self_attn.out_proj.bias": "attn_out.bias",
        "linear1.weight": "ffn_up.weight",
        "linear1.bias": "ffn_up.bias",
        "linear2.weight": "ffn_down.weight",
        "linear2.bias": "ffn_down.bias",
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
    }
    return f"qantara.vision.blk.{layer}.{suffixes[suffix]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flow-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flow_steps < 1:
        raise SystemExit("--flow-steps must be positive")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise SystemExit("checkpoint must contain config and model mappings")
    cfg = checkpoint["config"]
    _validate_config(cfg)
    stats = checkpoint.get("action_stats")
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise SystemExit("checkpoint is missing action_stats")
    action_mean = torch.as_tensor(stats["mean"])
    action_std = torch.as_tensor(stats["std"])
    expected = (int(cfg["action_dim"]),)
    if action_mean.shape != expected or action_std.shape != expected:
        raise SystemExit(f"action statistics must have shape {expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), arch=ARCH)
    _add_metadata(writer, cfg, action_mean, action_std, args.flow_steps)

    count = 0
    state = checkpoint["model"]
    _add_tensor(writer, "qantara.action_mean", action_mean)
    _add_tensor(writer, "qantara.action_std", action_std)
    count += 2

    encoder_bn_prefix = "encoder.projector.1"
    encoder_bn_scale = state[f"{encoder_bn_prefix}.weight"].float() / torch.sqrt(
        state[f"{encoder_bn_prefix}.running_var"].float() + 1.0e-5
    )
    encoder_bn_offset = (
        state[f"{encoder_bn_prefix}.bias"].float()
        - state[f"{encoder_bn_prefix}.running_mean"].float() * encoder_bn_scale
    )
    _add_tensor(writer, "qantara.encoder.proj_norm.scale", encoder_bn_scale)
    _add_tensor(writer, "qantara.encoder.proj_norm.offset", encoder_bn_offset)
    count += 2
    for source_name, tensor in state.items():
        if not source_name.startswith("encoder."):
            continue
        if source_name.endswith("num_batches_tracked"):
            continue
        if source_name.startswith(encoder_bn_prefix + "."):
            continue
        _add_tensor(writer, _encoder_name(source_name), tensor)
        count += 1

    bn_prefix = "predictor.state_head.net.1"
    bn_scale = state[f"{bn_prefix}.weight"].float() / torch.sqrt(
        state[f"{bn_prefix}.running_var"].float() + 1.0e-5
    )
    bn_offset = (
        state[f"{bn_prefix}.bias"].float()
        - state[f"{bn_prefix}.running_mean"].float() * bn_scale
    )
    _add_tensor(writer, "qantara.state_head.net.1.scale", bn_scale)
    _add_tensor(writer, "qantara.state_head.net.1.offset", bn_offset)
    count += 2
    for source_name, tensor in state.items():
        if not source_name.startswith("predictor."):
            continue
        if source_name.endswith("num_batches_tracked"):
            continue
        if source_name.startswith(bn_prefix + "."):
            continue
        destination = "qantara." + source_name.removeprefix("predictor.")
        _add_tensor(writer, destination, tensor)
        count += 1
    if count == 0:
        raise SystemExit("checkpoint contains no predictor tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {count} Qantara tensors to {args.output}")


if __name__ == "__main__":
    main()
