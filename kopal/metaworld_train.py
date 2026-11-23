"""KOPAL 的 MetaWorld 数据生成、训练和闭环评估入口。

本文件实现脚本专家数据采集、世界模型与策略的分阶段训练、checkpoint
保存与恢复，以及基于子目标点云轨迹的 MetaWorld 闭环评估。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import EncoderConfig, PlannerConfig, PolicyConfig, KOPALConfig, WorldModelConfig
from .dataset import (
    SlidingWindowTrajectoryDataset,
    collate_trajectory_samples,
    load_npz_trajectory_dataset,
)
from .metaworld_task import (
    ButtonObservation,
    METAWORLD_TASKS,
    MetaWorldButtonPressAdapter,
    collect_button_press_dataset,
)
from .system import KOPALSystem


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_training_root() -> Path:
    return repo_root() / "KOPAL_train"


def build_config(device: str) -> KOPALConfig:
    encoder = EncoderConfig(
        point_hidden_dims=(64, 128),
        token_dim=128,
        global_dim=256,
        abstract_points=(32, 12),
        knn=12,
    )
    object_encoder = EncoderConfig(
        point_hidden_dims=(64, 128),
        token_dim=96,
        global_dim=192,
        abstract_points=(16, 8),
        knn=12,
    )
    return KOPALConfig(
        object_count=1,
        object_points=64,
        scene_points=256,
        obstacle_points=0,
        device=device,
        world_model=WorldModelConfig(
            encoder=object_encoder,
            object_dim=256,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff_dim=512,
            jepa_dim=192,
        ),
        policy=PolicyConfig(
            scene_encoder=encoder,
            object_encoder=object_encoder,
            d_model=256,
            transformer_layers=4,
            transformer_heads=4,
            transformer_ff_dim=512,
            dropout=0.1,
            action_dim=4,
            action_horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            robot_state_dim=9,
            inference_steps=1,
        ),
        planner=PlannerConfig(
            horizon=16,
            num_keyframes=3,
            num_samples=32,
            num_elites=8,
            num_iterations=2,
            init_rotation_std=0.30,
            init_translation_std=0.035,
            min_translation_std=0.003,
            collision_weight=0.0,
            smoothness_weight=0.02,
            object_interaction_weight=0.0,
        ),
    )


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_dataset(path: Path, config: KOPALConfig) -> SlidingWindowTrajectoryDataset:
    tensors = load_npz_trajectory_dataset(path)
    if tensors.actions.shape[-1] != config.policy.action_dim:
        raise ValueError("Dataset action dimension does not match policy config")
    if tensors.robot_state.shape[-1] != config.policy.robot_state_dim:
        raise ValueError("Dataset state dimension does not match policy config")
    return SlidingWindowTrajectoryDataset(
        tensors,
        world_horizon=min(8, config.planner.horizon),
        policy_horizon=config.policy.action_horizon,
        n_obs_steps=config.policy.n_obs_steps,
        stride=1,
    )


def append_jsonl(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    system: KOPALSystem,
    config: KOPALConfig,
    policy_epoch: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "world_model": system.world_model.state_dict(),
            "policy": system.policy.state_dict(),
            "policy_ema": system.policy_ema.state_dict(),
            "policy_epoch": policy_epoch,
            "action_mean": action_mean.detach().cpu(),
            "action_std": action_std.detach().cpu(),
            "action_min": action_min.detach().cpu(),
            "action_max": action_max.detach().cpu(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    system: KOPALSystem,
    use_ema: bool = False,
) -> Dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    system.world_model.load_state_dict(checkpoint["world_model"])

    def load_policy_state(module, state):
        current = module.state_dict()
        compatible = {}
        for name, value in state.items():
            if name not in current:
                continue
            target = current[name]
            if target.shape == value.shape:
                compatible[name] = value
            elif name == "condition_type_proj.weight":
                merged = target.clone()
                rows = min(target.shape[0], value.shape[0])
                merged[:rows] = value[:rows]
                compatible[name] = merged
        module.load_state_dict(compatible, strict=False)

    load_policy_state(system.policy, checkpoint["policy"])
    if "policy_ema" in checkpoint:
        load_policy_state(system.policy_ema, checkpoint["policy_ema"])
        if use_ema:
            system.use_policy_ema()
    if "action_mean" in checkpoint:
        system.policy.set_action_stats(
            checkpoint["action_mean"].to(system.policy.action_mean.device),
            checkpoint["action_std"].to(system.policy.action_std.device),
        )
    if "action_min" in checkpoint and "action_max" in checkpoint:
        system.policy.set_action_bounds(
            checkpoint["action_min"].to(system.policy.action_min.device),
            checkpoint["action_max"].to(system.policy.action_max.device),
        )
    return checkpoint


def train(
    args: argparse.Namespace,
) -> None:
    device = args.device
    config = build_config(device)
    dataset = build_dataset(Path(args.dataset), config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_trajectory_samples,
        drop_last=False,
    )
    system = KOPALSystem(config).to(device)
    action_mean, action_std = dataset.action_stats
    action_min, action_max = dataset.action_bounds
    action_mean, action_std = action_mean.to(device), action_std.to(device)
    action_min, action_max = action_min.to(device), action_max.to(device)
    system.policy.set_action_stats(action_mean, action_std)
    system.policy.set_action_bounds(action_min, action_max)
    binary_mask, _, _ = dataset.action_binary_metadata
    system.policy.set_action_binary_mask(binary_mask.to(device))
    state_mean, state_std = dataset.state_stats
    system.policy.set_state_stats(
        state_mean.to(device),
        state_std.to(device),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "latest.pt"
    resume_epoch = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, system)
        resume_epoch = int(checkpoint.get("policy_epoch", 0))
        print(f"resuming policy epoch {resume_epoch}")

    world_optimizer = torch.optim.AdamW(
        system.world_model.parameters(),
        lr=args.world_lr,
        weight_decay=1.0e-5,
    )
    policy_optimizer = torch.optim.AdamW(
        system.policy.parameters(),
        lr=args.policy_lr,
        weight_decay=args.policy_weight_decay,
    )

    start = time.time()
    resume_world = (
        args.resume
        and resume_epoch > 0
        and (output_dir / "world.pt").exists()
    )
    if args.world_epochs > 0 and not resume_world:
        for epoch in range(1, args.world_epochs + 1):
            system.world_model.set_epoch(epoch)
            totals = []
            component_totals: Dict[str, list[float]] = {}
            progress = tqdm(loader, desc=f"world {epoch}/{args.world_epochs}")
            for batch in progress:
                batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
                losses = system.train_world_step(
                    world_optimizer,
                    batch,
                    grad_clip=args.grad_clip,
                )
                totals.append(losses["wm_total_loss"])
                for name, value in losses.items():
                    component_totals.setdefault(name, []).append(value)
                progress.set_postfix(loss=f"{totals[-1]:.5f}")
            record = {
                "stage": "world",
                "epoch": epoch,
                "loss": float(np.mean(totals)),
                "elapsed_sec": time.time() - start,
            }
            record.update(
                {
                    name: float(np.mean(values))
                    for name, values in component_totals.items()
                }
            )
            append_jsonl(metrics_path, record)
            print(record)
        torch.save(
            {"world_model": system.world_model.state_dict(), "config": config.to_dict()},
            output_dir / "world.pt",
        )

    for epoch in range(resume_epoch + 1, args.policy_epochs + 1):
        totals = []
        progress = tqdm(
            loader,
            desc=f"policy {epoch}/{args.policy_epochs}",
        )
        for batch in progress:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            losses = system.train_policy_step(
                policy_optimizer,
                batch,
                grad_clip=args.grad_clip,
            )
            totals.append(losses["policy_total_loss"])
            progress.set_postfix(loss=f"{totals[-1]:.5f}")
        record = {
            "stage": "policy",
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "elapsed_sec": time.time() - start,
        }
        append_jsonl(metrics_path, record)
        print(record)
        if epoch % args.save_every == 0 or epoch == args.policy_epochs:
            save_checkpoint(
                checkpoint_path,
                system,
                config,
                epoch,
                action_mean,
                action_std,
                action_min,
                action_max,
            )
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                system,
                config,
                epoch,
                action_mean,
                action_std,
                action_min,
                action_max,
            )


def _policy_observation(adapter: MetaWorldButtonPressAdapter):
    observation = adapter.observe()
    return observation


def _move_observation(observation: ButtonObservation, device: str) -> ButtonObservation:
    observation.scene_points = observation.scene_points.to(device)
    observation.object_points = observation.object_points.to(device)
    observation.robot_state = observation.robot_state.to(device)
    observation.goal_objects = observation.goal_objects.to(device)
    observation.object_center = observation.object_center.to(device)
    observation.target_position = observation.target_position.to(device)
    return observation


def _update_subgoal(
    system: KOPALSystem,
    observation: ButtonObservation,
    approach_threshold: float,
    horizon: int,
    object_points: int,
    device: str,
    requires_grasp: bool,
    grasp_width_threshold: float,
) -> float:
    distance = torch.linalg.vector_norm(
        observation.robot_state[0, :3] - observation.object_center[0]
    )
    object_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    point_mask = torch.ones(
        1,
        1,
        object_points,
        dtype=torch.bool,
        device=device,
    )
    gripper_width = None
    if observation.robot_state.shape[-1] >= 9:
        gripper_width = torch.linalg.vector_norm(
            observation.robot_state[0, 3:6]
            - observation.robot_state[0, 6:9]
        )
    grasp_ready = (
        not requires_grasp
        or (
            gripper_width is not None
            and float(gripper_width) <= grasp_width_threshold
        )
    )
    if float(distance) > approach_threshold or not grasp_ready:
        # 演示策略在夹爪接近阶段学习静止物体轨迹。
        # 零动作世界模型推演正对应该轨迹，因此直接生成静态子目标，
        # 将交叉熵规划保留给接触与操作阶段。
        trajectory = observation.object_points[:, None].expand(
            -1,
            horizon,
            -1,
            -1,
            -1,
        )
        system.planner.reset()
        system.subgoal_buffer.update(
            trajectory,
            current_points=observation.object_points,
            object_mask=object_mask,
            point_mask=point_mask,
        )
    else:
        system.update_subgoal(
            observation.object_points,
            observation.goal_objects,
            object_mask=object_mask,
            point_mask=point_mask,
        )
    return float(distance)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = args.device
    config = build_config(device)
    system = KOPALSystem(config).to(device)
    checkpoint = load_checkpoint(
        Path(args.checkpoint),
        system,
        use_ema=args.use_policy_ema,
    )
    system.world_model.eval()
    system.policy.eval()
    adapter = MetaWorldButtonPressAdapter(
        seed=args.eval_seed,
        scene_points=config.scene_points,
        object_points=config.object_points,
        max_episode_steps=args.eval_max_steps,
        stop_on_success=True,
        task_name=args.task_name,
    )
    successes = []
    rewards = []
    lengths = []
    action_norms = []
    try:
        for episode in range(args.eval_episodes):
            episode_seed = args.eval_seed + episode
            seed_everything(episode_seed)
            observation = _move_observation(
                adapter.reset(episode_seed, compute_goal=True),
                device,
            )
            system.planner.reset()
            system.subgoal_buffer.clear()
            _update_subgoal(
                system,
                observation,
                args.approach_threshold,
                config.planner.horizon,
                config.object_points,
                device,
                requires_grasp=args.task_name in {
                    "pick-place-v3",
                    "peg-insert-side-v3",
                    "assembly-v3",
                },
                grasp_width_threshold=args.grasp_width_threshold,
            )
            scene_history = [observation.scene_points[0], observation.scene_points[0]]
            state_history = [observation.robot_state[0], observation.robot_state[0]]
            success = False
            total_reward = 0.0
            steps = 0
            for _ in range(args.eval_max_steps):
                if steps % args.replan_every == 0:
                    current_obs = _move_observation(adapter.observe(), device)
                    _update_subgoal(
                        system,
                        current_obs,
                        args.approach_threshold,
                        config.planner.horizon,
                        config.object_points,
                        device,
                        requires_grasp=args.task_name in {
                            "pick-place-v3",
                            "peg-insert-side-v3",
                            "assembly-v3",
                        },
                        grasp_width_threshold=args.grasp_width_threshold,
                    )
                prediction = system.act(
                    torch.stack(scene_history[-2:]).unsqueeze(0).to(device),
                    torch.stack(state_history[-2:]).unsqueeze(0).to(device),
                    num_steps=1,
                )
                chunk = prediction["action"][0].clamp(-1.0, 1.0)
                executed = 0
                for policy_action in chunk[: config.policy.n_action_steps]:
                    action = policy_action.detach().cpu().numpy()
                    _, reward, terminated, truncated, info = adapter.step(action)
                    executed += 1
                    steps += 1
                    total_reward += float(reward)
                    action_norms.append(float(np.linalg.norm(action)))
                    next_observation = _move_observation(adapter.observe(), device)
                    scene_history.append(next_observation.scene_points[0])
                    state_history.append(next_observation.robot_state[0])
                    success = success or float(info.get("success", 0.0)) > 0.5
                    if success or terminated or truncated:
                        break
                system.subgoal_buffer.advance(executed)
                if success or terminated or truncated:
                    break
            successes.append(float(success))
            rewards.append(total_reward)
            lengths.append(steps)
            print(
                f"eval episode={episode:03d} success={int(success)} "
                f"steps={steps} reward={total_reward:.3f}"
            )
    finally:
        adapter.close()
    result = {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "episodes": int(args.eval_episodes),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "policy_epoch": int(checkpoint.get("policy_epoch", -1)),
    }
    output_path = Path(args.output_dir) / "eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    training_root = default_training_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("generate", "train", "eval"), required=True)
    parser.add_argument(
        "--task-name",
        choices=METAWORLD_TASKS,
        default="reach-v3",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--world-epochs", type=int, default=100)
    parser.add_argument("--policy-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--world-lr", type=float, default=1.0e-4)
    parser.add_argument("--policy-lr", type=float, default=1.0e-4)
    parser.add_argument("--policy-weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--approach-threshold", type=float, default=0.10)
    parser.add_argument("--grasp-width-threshold", type=float, default=0.06)
    parser.add_argument("--use-policy-ema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_root = default_training_root()
    task_slug = args.task_name.replace("-v3", "").replace("-", "_")
    if args.dataset is None:
        args.dataset = str(training_root / "datasets" / f"{task_slug}.npz")
    if args.output_dir is None:
        args.output_dir = str(training_root / "outputs" / task_slug)
    seed_everything(args.seed)
    if args.mode == "generate":
        metadata = collect_button_press_dataset(
            args.dataset,
            episodes=args.episodes,
            seed=args.seed,
            task_name=args.task_name,
        )
        print(json.dumps(metadata, indent=2))
    elif args.mode == "train":
        train(args)
    else:
        args.checkpoint = args.checkpoint or str(Path(args.output_dir) / "latest.pt")
        evaluate(args)


if __name__ == "__main__":
    main()

