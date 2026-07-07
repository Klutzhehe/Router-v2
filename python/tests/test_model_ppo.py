"""Smoke tests: model forward/act/evaluate and one PPO update cycle."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_router.config import EnvConfig                      # noqa: E402
from pcb_router.env import RoutingEnv                        # noqa: E402
from pcb_router.generator import generate_board              # noqa: E402
from pcb_router.model import DualStreamRouter, RouterAction  # noqa: E402
from pcb_router.ppo import PPO, PPOConfig, collect_rollout, to_torch  # noqa: E402


def test_model_forward(seed=5):
    torch.manual_seed(seed)
    env = RoutingEnv(lambda r: generate_board(0, r), seed=seed)
    obs, masks = env.reset()
    model = DualStreamRouter()
    t_obs = to_torch(obs, "cpu")
    t_masks = {k: torch.from_numpy(v).unsqueeze(0) for k, v in masks.items()}

    action, logp, value = model.act(t_obs, t_masks)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()
    # sampled action must respect the mask
    assert masks["type"][int(action.action_type)] == 1
    if int(action.action_type) == 0:
        assert masks["angle"][int(action.angle_bin)] == 1

    logp2, ent, v2 = model.evaluate_actions(
        t_obs, t_masks, RouterAction(*(x for x in action)))
    assert torch.isfinite(logp2).all() and torch.isfinite(ent).all()
    assert torch.allclose(logp, logp2, atol=1e-5), "act/evaluate logp mismatch"
    print("test_model_forward OK")


def test_ppo_update(seed=6):
    torch.manual_seed(seed)
    env = RoutingEnv(lambda r: generate_board(0, r), cfg=EnvConfig(), seed=seed)
    model = DualStreamRouter()
    ppo = PPO(model, PPOConfig(epochs=1, minibatch=64), device="cpu")
    buf, stats, _ = collect_rollout(env, model, 256, "cpu")
    out = ppo.update(buf)
    assert all(np.isfinite(v) for v in out.values()), out
    print(f"test_ppo_update OK ({out})")


if __name__ == "__main__":
    test_model_forward()
    test_ppo_update()
    print("model/ppo: all tests passed")
