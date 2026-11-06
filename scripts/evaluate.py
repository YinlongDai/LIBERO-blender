"""Closed-loop LIBERO eval of a policy, with MuJoCo or Blender images and fixed or random lighting.

    python scripts/evaluate.py --render configs/render/eevee.yaml --policy-port 6000 --name my_run
    python scripts/evaluate.py --render configs/render/eevee.yaml --lighting configs/lighting/random.yaml ...
    python scripts/evaluate.py --render mujoco --task-id 3 ...

The protocol (tasks, trials, seed, settle steps, replanning, videos) is in configs/eval.yaml. Writes
<results>/eval/<name>/<suite>/task<id>.json with every trial's success, length and lighting, and with
save_videos, videos/task<id>_trial<n>.mp4.

The policy runs as a server in its own process and environment (see scripts/policies/). Each query
sends, in the format of libero_blender/protocol.py,
    {"cmd": "act", "language": str, "seed": int} + agentview_image, robot0_eye_in_hand_image (uint8, raw
    robosuite orientation), robot0_joint_pos, robot0_gripper_qpos, robot0_eef_pos, robot0_eef_quat
and expects {"ok": true} + actions (float32 [N, 7]), of which `replan_steps` are executed. `seed` is
fixed per trial, for policies that sample.
"""
import argparse
import json
import time
from pathlib import Path

import imageio
import numpy as np
import yaml

from libero_blender.env import BlenderEnv
from libero_blender.paths import RESULTS, ROOT
from libero_blender.protocol import connect, recv_message, send_message
from libero_blender.scene import make_env, num_tasks

OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos",
            "robot0_eef_pos", "robot0_eef_quat"]


class MujocoEnv:
    """The plain LIBERO env with BlenderEnv's call signatures, for evals with MuJoCo images."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, lighting_seed=None, render_now=True):
        return self.env.reset()

    def set_init_state(self, init_state, render_now=True):
        return self.env.set_init_state(init_state)

    def step(self, action, render_now=True):
        return self.env.step(action)

    def render_camera(self, camera, size):
        # MuJoCo's offscreen buffer is 640 x 480, so size can be at most 480.
        return self.env.sim.render(size, size, camera_name=camera)[::-1]


class PolicyClient:
    def __init__(self, port):
        self.sock = connect(port)

    def act(self, obs, language, seed):
        send_message(self.sock, {"cmd": "act", "language": language, "seed": seed}, {key: obs[key] for key in OBS_KEYS})
        reply, arrays = recv_message(self.sock)
        assert reply["ok"], reply.get("error")
        return arrays["actions"]


def run_trial(env, policy, init_state, language, protocol, max_steps, seed, lighting_seed, frames=None):
    """Runs one trial; with `frames`, appends an agentview image of every step to it."""
    env.seed(protocol["seed"])
    env.reset(lighting_seed=lighting_seed, render_now=False)
    env.set_init_state(init_state, render_now=False)
    for i in range(protocol["settle_steps"]):
        obs, _, _, _ = env.step(protocol["settle_action"], render_now=(i == protocol["settle_steps"] - 1))

    steps = 0
    while steps < max_steps:
        actions = policy.act(obs, language, seed)
        num_actions = min(protocol["replan_steps"], len(actions), max_steps - steps)
        for k in range(num_actions):
            # Images are only needed for the observation the next query sees.
            obs, _, done, _ = env.step(actions[k].tolist(), render_now=(k == num_actions - 1))
            if frames is not None:
                frames.append(env.render_camera("agentview", protocol["video_size"]))
            steps += 1
            if done or env.check_success():
                return True, steps
    return False, max_steps


def evaluate_task(task_id, args, protocol, lighting, policy, out):
    suite = protocol["suite"]
    env, task, init_states = make_env(suite, task_id, protocol["image_size"], seed=protocol["seed"])
    assert protocol["trials_per_task"] <= len(init_states), f"{suite} has {len(init_states)} init states per task"
    if args.render == "mujoco":
        assert not args.lighting, "random lighting needs Blender images"
        env, render_config = MujocoEnv(env), None
    else:
        env.set_init_state(init_states[0])  # the key light hangs above the scene as it starts
        env = BlenderEnv(env, args.render, lighting=args.lighting, image_size=protocol["image_size"])
        render_config = env.renderer.tag

    start = time.time()
    successes, lengths = [], []
    for trial in range(protocol["trials_per_task"]):
        seed = protocol["seed"] * 100003 + task_id * 101 + trial
        frames = [] if protocol.get("save_videos") else None
        # Lighting depends on (seed, task, trial) only, so every policy sees the same lightings.
        success, length = run_trial(env, policy, init_states[trial], task.language, protocol,
                                    protocol["max_steps"][suite], seed, [protocol["seed"], task_id, trial], frames)
        successes.append(success)
        lengths.append(length)
        if frames:
            (out / "videos").mkdir(exist_ok=True)
            imageio.mimwrite(out / "videos" / f"task{task_id}_trial{trial}.mp4", frames, fps=20)  # LIBERO runs at 20 Hz
        print(f"task {task_id} trial {trial}: {'success' if success else 'failure'} in {length} steps", flush=True)

    result = {
        "suite": suite, "task_id": task_id, "task": task.name, "language": task.language,
        "render_config": render_config, "lighting": lighting, "trial_lightings": env.lightings if lighting else [],
        "policy": args.policy, "protocol": protocol,
        "successes": successes, "lengths": lengths, "success_rate": float(np.mean(successes)),
        "wall_seconds": round(time.time() - start, 1),
    }
    (out / f"task{task_id}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("task", "success_rate", "wall_seconds")}), flush=True)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", required=True, help="configs/render/*.yaml, or mujoco")
    parser.add_argument("--lighting", help="configs/lighting/*.yaml: random lighting per trial")
    parser.add_argument("--task-id", type=int, help="evaluate only this task, in place of the protocol's tasks")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--policy", default="", help="what the policy server runs, recorded in the results")
    parser.add_argument("--name", required=True)
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "eval.yaml"))
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    lighting = yaml.safe_load(Path(args.lighting).read_text()) if args.lighting else None
    out = RESULTS / "eval" / args.name / protocol["suite"]
    out.mkdir(parents=True, exist_ok=True)

    if args.task_id is not None:
        task_ids = [args.task_id]
    elif protocol.get("tasks", "all") == "all":
        task_ids = range(num_tasks(protocol["suite"]))
    else:
        task_ids = protocol["tasks"]
    policy = PolicyClient(args.policy_port)
    for task_id in task_ids:
        evaluate_task(task_id, args, protocol, lighting, policy, out)


if __name__ == "__main__":
    main()
