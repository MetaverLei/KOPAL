"""KOPAL 分阶段命令行训练入口。

本文件支持分别训练世界模型和 MeanFlow 策略，可读取本地 NPZ 轨迹数据，
也可在没有外部数据时构建合成轨迹完成训练流程检查。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .config import KOPALConfig
from .dataset import (
    SlidingWindowTrajectoryDataset,
    collate_trajectory_samples,
    load_npz_trajectory_dataset,
    make_synthetic_dataset,
)
from .system import KOPALSystem


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args: argparse.Namespace, config: KOPALConfig) -> SlidingWindowTrajectoryDataset:
    if args.data is None:
        tensors = make_synthetic_dataset(
            objects=config.object_count,
            object_points=config.object_points,
            scene_points=config.scene_points,
            state_dim=config.policy.robot_state_dim,
            action_dim=config.policy.action_dim,
            seed=args.seed,
        )
    else:
        tensors = load_npz_trajectory_dataset(args.data)
    return SlidingWindowTrajectoryDataset(
        tensors,
        world_horizon=config.planner.horizon,
        policy_horizon=config.policy.action_horizon,
        n_obs_steps=config.policy.n_obs_steps,
    )


def stage_train(args: argparse.Namespace) -> Dict[str, float]:
    seed_everything(args.seed)
    config = KOPALConfig(device=args.device)
    dataset = build_dataset(args, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_trajectory_samples,
    )
    system = KOPALSystem(config).to(args.device)
    model = system.world_model if args.stage == "world" else system.policy
    if args.stage == "world":
        # 基于步数的训练器没有 epoch 循环，将首个逻辑轮次设为 1，
        # 使默认的零轮 warmup 在第一次更新时即启用 JEPA 目标。
        system.world_model.set_epoch(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    if args.stage == "policy":
        mean, std = dataset.action_stats
        system.policy.set_action_stats(mean.to(args.device), std.to(args.device))
    losses: Dict[str, float] = {}
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = {name: value.to(args.device) for name, value in batch.items()}
            if args.stage == "world":
                losses = system.train_world_step(optimizer, batch)
            else:
                losses = system.train_policy_step(optimizer, batch)
            step += 1
            if step % max(args.log_every, 1) == 0:
                print(step, losses)
            if step >= args.steps:
                break
    if args.output:
        checkpoint = {
            "config": config.to_dict(),
            "world_model": system.world_model.state_dict(),
            "policy": system.policy.state_dict(),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output)
        print(f"saved checkpoint to {output}")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("world", "policy"), required=True)
    parser.add_argument("--data", type=str, default=None, help="Optional NPZ trajectory tensor file")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    stage_train(parser.parse_args())


if __name__ == "__main__":
    main()

