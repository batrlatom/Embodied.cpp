#!/usr/bin/env python3
"""Create a deterministic Qantara predictor input and optional parity report."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--qantara-repo", type=Path, required=True)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--gguf", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.qantara_repo.resolve()))
    from qantara import WorldModel
    from qantara.training import load_checkpoint

    torch.set_num_threads(1)
    model = WorldModel.from_checkpoint(load_checkpoint(args.checkpoint)).eval()
    cfg = model.cfg
    if cfg.camera_names != ("agentview",) or cfg.task_names or cfg.robotttt:
        raise SystemExit("minimal parity exporter needs a dense single-camera checkpoint")
    generator = torch.Generator().manual_seed(args.seed)
    history = cfg.num_frames - 1
    latents = torch.randn(history, cfg.latent_dim, generator=generator)
    actions = torch.randn(history - 1, cfg.action_block_dim, generator=generator)
    video_noise = torch.randn(cfg.action_block_dim, generator=generator)
    action_noise = torch.randn(cfg.action_block_dim, generator=generator)

    z = latents.unsqueeze(0)
    a = actions.unsqueeze(0)
    with torch.no_grad():
        _, next_z = model.predictor._query(
            z, a, video_noise.unsqueeze(0), z[:, -1], 0.0, 0.0
        )
        current = action_noise.unsqueeze(0)
        for step in range(args.flow_steps):
            velocity, _ = model.predictor._query(
                z,
                a,
                current,
                next_z,
                step / args.flow_steps,
                1.0,
            )
            current = current + velocity / args.flow_steps
    reference = current[0].float().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(
            struct.pack(
                "<4I",
                0x51545241,
                history,
                cfg.latent_dim,
                cfg.action_block_dim,
            )
        )
        for value in (latents, actions, video_noise, action_noise):
            stream.write(value.float().contiguous().numpy().tobytes())
    reference_path = args.output.with_suffix(args.output.suffix + ".json")
    reference_path.write_text(
        json.dumps({"action": reference.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} and {reference_path}")

    if args.executable or args.gguf:
        if not args.executable or not args.gguf:
            raise SystemExit("--executable and --gguf must be supplied together")
        completed = subprocess.run(
            [str(args.executable), str(args.gguf), str(args.output)],
            check=True,
            text=True,
            capture_output=True,
        )
        print(completed.stderr, end="", file=sys.stderr)
        print(completed.stdout, end="")
        action_line = next(
            line for line in completed.stdout.splitlines() if line.startswith("action ")
        )
        actual = np.asarray(
            [float(value) for value in action_line.split()[1:]], dtype=np.float32
        )
        difference = np.abs(actual - reference)
        print(
            "parity "
            f"max_abs={difference.max():.8g} "
            f"mean_abs={difference.mean():.8g} "
            f"cosine={np.dot(actual, reference) / (np.linalg.norm(actual) * np.linalg.norm(reference)):.9f}"
        )


if __name__ == "__main__":
    main()
