"""Example policy server for scripts/evaluate.py: a semantic_diffusion MoDE checkpoint.

Runs in semantic_diffusion's own environment (Python 3.11 + torch):
    SEMANTIC_DIFFUSION=<semantic_diffusion>/src/diffusion_policy
    $SEMANTIC_DIFFUSION/.venv/bin/python scripts/policies/mode_server.py --run-dir <run dir> --ckpt 60000 --port 6000

Preprocessing follows semantic_diffusion's LIBERO eval: both images rotated 180 degrees, state = 7 joint
positions + 2 gripper finger positions, 10 denoising steps, gripper action binarized by sign. The
sampling noise comes from one torch.Generator per trial seed.
"""
import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), os.environ["SEMANTIC_DIFFUSION"]]
from libero_env.eval_libero import Policy  # noqa: E402
from libero_blender.protocol import recv_message, send_message  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default="60000")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--inference-steps", type=int, default=10)
    args = parser.parse_args()
    policy = Policy(Path(args.run_dir), args.ckpt, device="cuda")
    generator, generator_seed = None, None

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(1)
    print(f"MoDE policy server listening on 127.0.0.1:{args.port}", flush=True)
    connection, _ = listener.accept()
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        try:
            header, obs = recv_message(connection)
        except ConnectionError:
            return
        if header["seed"] != generator_seed:
            generator_seed = header["seed"]
            generator = torch.Generator().manual_seed(generator_seed)
        images = {"static": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])[None, None],
                  "gripper": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])[None, None]}
        states = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32)[None, None]
        actions = policy(images, states, header["language"], args.inference_steps, generator)[0]
        actions[:, -1] = np.where(actions[:, -1] < 0, -1.0, 1.0)
        send_message(connection, {"ok": True}, {"actions": actions.astype(np.float32)})


if __name__ == "__main__":
    main()
