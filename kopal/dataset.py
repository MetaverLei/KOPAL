"""KOPAL 轨迹数据格式与滑动窗口数据集。

本文件定义场景点云、关键物体固定对应点轨迹、机器人状态和动作的联合数据
格式，并实现滑动窗口切分、NPZ 读写和用于接口验证的确定性合成轨迹生成。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .geometry import apply_object_action, interpolate_rigid, se3_exp

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class TrajectoryTensors:
    scene_points: torch.Tensor       # 场景点云，形状为 (E,T,Ns,3)
    object_points: torch.Tensor      # 关键物体点云，形状为 (E,T,O,No,3)
    robot_state: torch.Tensor        # 机器人状态，形状为 (E,T,Ds)
    actions: torch.Tensor            # 连续动作，形状为 (E,T,A)
    object_mask: Optional[torch.Tensor] = None
    point_mask: Optional[torch.Tensor] = None
    episode_lengths: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.scene_points.ndim != 4 or self.scene_points.shape[-1] != 3:
            raise ValueError("scene_points must be (E,T,Ns,3)")
        if self.object_points.ndim != 5 or self.object_points.shape[-1] != 3:
            raise ValueError("object_points must be (E,T,O,No,3)")
        if self.robot_state.ndim != 3 or self.actions.ndim != 3:
            raise ValueError("robot_state and actions must both be rank-3")
        episodes, steps = self.scene_points.shape[:2]
        if self.object_points.shape[:2] != (episodes, steps):
            raise ValueError("scene_points and object_points must share (E,T)")
        if self.robot_state.shape[:2] != (episodes, steps) or self.actions.shape[:2] != (episodes, steps):
            raise ValueError("robot_state and actions must share (E,T) with scene_points")
        objects, object_points = self.object_points.shape[2:4]
        if self.object_mask is not None and self.object_mask.shape != (episodes, steps, objects):
            raise ValueError("object_mask has an inconsistent shape")
        if self.point_mask is not None and self.point_mask.shape != (episodes, steps, objects, object_points):
            raise ValueError("point_mask has an inconsistent shape")
        if self.episode_lengths is not None:
            if self.episode_lengths.shape != (episodes,):
                raise ValueError("episode_lengths must have shape (E,)")
            if ((self.episode_lengths <= 0) | (self.episode_lengths > steps)).any():
                raise ValueError("episode_lengths must be within the padded trajectory length")

    def to(self, device: str | torch.device) -> "TrajectoryTensors":
        return TrajectoryTensors(
            self.scene_points.to(device),
            self.object_points.to(device),
            self.robot_state.to(device),
            self.actions.to(device),
            self.object_mask.to(device) if self.object_mask is not None else None,
            self.point_mask.to(device) if self.point_mask is not None else None,
            self.episode_lengths.to(device) if self.episode_lengths is not None else None,
        )


class SlidingWindowTrajectoryDataset(Dataset):
    """按轨迹时间轴切分训练窗口。

    每个样本同时提供世界模型训练所需的状态转移，以及策略训练所需的未来
    关键物体轨迹条件和专家动作块。所有窗口保持同一固定点对应关系。
    """

    def __init__(
        self,
        tensors: TrajectoryTensors,
        world_horizon: int = 4,
        policy_horizon: int = 8,
        n_obs_steps: int = 2,
        stride: int = 1,
    ) -> None:
        self.tensors = tensors
        self.world_horizon = world_horizon
        self.policy_horizon = policy_horizon
        self.n_obs_steps = n_obs_steps
        self.stride = stride
        required_length = max(world_horizon, policy_horizon) + n_obs_steps
        if tensors.scene_points.shape[1] < required_length:
            raise ValueError("Trajectories are too short for the requested horizons")
        self._index: list[tuple[int, int]] = []
        for episode in range(tensors.scene_points.shape[0]):
            episode_length = (
                int(tensors.episode_lengths[episode])
                if tensors.episode_lengths is not None
                else tensors.scene_points.shape[1]
            )
            for start in range(
                n_obs_steps - 1,
                episode_length - max(world_horizon, policy_horizon),
                stride,
            ):
                self._index.append((episode, start))

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _gather_future(values: torch.Tensor, start: int, horizon: int) -> torch.Tensor:
        end = min(start + horizon, values.shape[0])
        future = values[start:end]
        if future.shape[0] < horizon:
            pad = values[-1:].expand(horizon - future.shape[0], *values.shape[1:])
            future = torch.cat((future, pad), dim=0)
        return future

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode, start = self._index[index]
        data = self.tensors
        obs_start = start - self.n_obs_steps + 1
        scene = data.scene_points[episode, obs_start : start + 1]
        robot_state = data.robot_state[episode, obs_start : start + 1]
        current_objects = self._gather_future(
            data.object_points[episode], start, self.world_horizon
        )
        target_objects = self._gather_future(
            data.object_points[episode], start + 1, self.world_horizon
        )
        subgoal = self._gather_future(
            data.object_points[episode], start + 1, self.policy_horizon
        )
        actions = self._gather_future(data.actions[episode], start, self.policy_horizon)
        output = {
            "scene_points": scene,
            "robot_state": robot_state,
            "current_objects": current_objects,
            "target_objects": target_objects,
            "subgoal_trajectory": subgoal,
            "actions": actions,
        }
        if data.object_mask is not None:
            output["object_mask"] = self._gather_future(data.object_mask[episode], start, self.world_horizon)
            output["subgoal_mask"] = self._gather_future(
                data.object_mask[episode], start + 1, self.policy_horizon
            )
        if data.point_mask is not None:
            output["point_mask"] = self._gather_future(
                data.point_mask[episode], start, self.world_horizon
            )
            output["subgoal_point_mask"] = self._gather_future(
                data.point_mask[episode], start + 1, self.policy_horizon
            )
        return output

    @property
    def action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self._valid_actions()
        return flat.mean(dim=0), flat.std(dim=0, unbiased=False).clamp_min(1.0e-4)

    @property
    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self._valid_actions()
        return flat.min(dim=0).values, flat.max(dim=0).values

    def _valid_actions(self) -> torch.Tensor:
        actions = self.tensors.actions
        if self.tensors.episode_lengths is None:
            return actions.reshape(-1, actions.shape[-1])
        valid = torch.cat(
            [
                actions[index, : int(length)]
                for index, length in enumerate(self.tensors.episode_lengths)
            ],
            dim=0,
        )
        return valid.reshape(-1, actions.shape[-1])

    @property
    def state_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.tensors.robot_state
        if self.tensors.episode_lengths is None:
            flat = states.reshape(-1, states.shape[-1])
        else:
            flat = torch.cat(
                [
                    states[index, : int(length)]
                    for index, length in enumerate(
                        self.tensors.episode_lengths
                    )
                ],
                dim=0,
            ).reshape(-1, states.shape[-1])
        return (
            flat.mean(dim=0),
            flat.std(dim=0, unbiased=False).clamp_min(1.0e-4),
        )

    @property
    def action_binary_metadata(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self._valid_actions()
        minimum, maximum = flat.min(dim=0).values, flat.max(dim=0).values
        is_binary = torch.zeros_like(minimum, dtype=torch.bool)
        for dimension in range(flat.shape[-1]):
            values = torch.unique(flat[:, dimension].round(decimals=5))
            is_binary[dimension] = values.numel() <= 2
        return is_binary, minimum, maximum


def collate_trajectory_samples(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = samples[0].keys()
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}


def load_npz_trajectory_dataset(path: str | Path) -> TrajectoryTensors:
    """读取符合 KOPAL 数据格式的 NPZ 或 PyTorch 文件。"""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        if np is None:
            raise RuntimeError("NumPy is required to load .npz datasets")
        with np.load(path) as archive:
            arrays = {name: torch.as_tensor(archive[name]) for name in archive.files}
    else:
        arrays = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(arrays, dict):
        raise ValueError("Dataset file must contain a dictionary of tensors")
    required = ("scene_points", "object_points", "robot_state", "actions")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"Dataset is missing keys: {missing}")
    return TrajectoryTensors(
        **{name: torch.as_tensor(arrays[name]).float() for name in required},
        object_mask=torch.as_tensor(arrays["object_mask"]).bool() if "object_mask" in arrays else None,
        point_mask=torch.as_tensor(arrays["point_mask"]).bool() if "point_mask" in arrays else None,
        episode_lengths=(
            torch.as_tensor(arrays["episode_lengths"]).long()
            if "episode_lengths" in arrays
            else None
        ),
    )


def save_npz_trajectory_dataset(tensors: TrajectoryTensors, path: str | Path) -> None:
    """将轨迹数据保存为可移植的压缩 NPZ 文件。"""
    if np is None:
        raise RuntimeError("NumPy is required to save .npz datasets")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scene_points": tensors.scene_points.detach().cpu().numpy(),
        "object_points": tensors.object_points.detach().cpu().numpy(),
        "robot_state": tensors.robot_state.detach().cpu().numpy(),
        "actions": tensors.actions.detach().cpu().numpy(),
    }
    if tensors.object_mask is not None:
        arrays["object_mask"] = tensors.object_mask.detach().cpu().numpy()
    if tensors.point_mask is not None:
        arrays["point_mask"] = tensors.point_mask.detach().cpu().numpy()
    if tensors.episode_lengths is not None:
        arrays["episode_lengths"] = tensors.episode_lengths.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def make_synthetic_dataset(
    episodes: int = 4,
    steps: int = 24,
    objects: int = 2,
    object_points: int = 32,
    scene_points: int = 96,
    state_dim: int = 9,
    action_dim: int = 7,
    seed: int = 7,
) -> TrajectoryTensors:
    """生成固定点对应的小规模刚体运动数据，用于检查训练和推理接口。"""
    if state_dim < 9:
        raise ValueError("state_dim must be at least 9 for the synthetic embodiment")
    if action_dim < 7:
        raise ValueError("action_dim must be at least 7 for the synthetic embodiment")
    generator = torch.Generator().manual_seed(seed)
    object_clouds = torch.randn(objects, object_points, 3, generator=generator) * 0.08
    default_offsets = torch.zeros(objects, 3)
    if objects > 1:
        default_offsets[1] = torch.tensor([0.28, 0.05, 0.0])
    object_clouds += default_offsets.unsqueeze(1)
    background = torch.randn(scene_points * 2, 3, generator=generator) * 0.35
    background[:, 2] = background[:, 2].abs() * 0.08
    background += torch.tensor([0.12, -0.02, 0.08])

    all_scene = []
    all_objects = []
    all_state = []
    all_actions = []
    for episode in range(episodes):
        start_translation = torch.empty(objects, 3).uniform_(-0.05, 0.05, generator=generator)
        goal_translation = torch.empty(objects, 3).uniform_(-0.10, 0.10, generator=generator)
        goal_translation[0, 0] += 0.15
        start_rotation, _ = se3_exp(
            torch.cat(
                (
                    torch.empty(objects, 3).uniform_(-0.15, 0.15, generator=generator),
                    torch.zeros(objects, 3),
                ),
                dim=-1,
            )
        )
        goal_rotation, _ = se3_exp(
            torch.cat(
                (
                    torch.empty(objects, 3).uniform_(-0.15, 0.15, generator=generator),
                    torch.zeros(objects, 3),
                ),
                dim=-1,
            )
        )
        episode_objects = []
        episode_scene = []
        episode_state = []
        episode_actions = []
        previous_centroids = None
        for step_index in range(steps):
            alpha = torch.tensor([step_index / max(steps - 1, 1)])
            rotation, translation = interpolate_rigid(
                goal_rotation @ start_rotation.transpose(-1, -2),
                goal_translation - start_translation,
                alpha,
            )
            rotation = rotation[..., 0, :, :]
            translation = translation[..., 0, :]
            moved = apply_object_action(
                object_clouds,
                rotation,
                translation + start_translation,
            )
            centroids = moved.mean(dim=-2)
            if previous_centroids is None:
                centroid_delta = centroids - moved.mean(dim=-2)
            else:
                centroid_delta = centroids - previous_centroids
            previous_centroids = centroids.detach()
            flat_objects = moved.reshape(-1, 3)
            scene = torch.cat((background, flat_objects), dim=0)
            scene_sampled = scene[torch.linspace(0, scene.shape[0] - 1, scene_points).long()]
            eef = centroids[0] + torch.tensor([0.02, 0.0, 0.08])
            quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
            state = torch.cat((eef, quaternion, torch.tensor([0.8, 0.2])))
            action = torch.cat(
                (
                    centroid_delta[0] * 4.0,
                    rotation[0, 2, 1:2] * 0.2,
                    rotation[0, 0, 2:3] * 0.2,
                    rotation[0, 1, 0:1] * 0.2,
                    torch.tensor([0.7]),
                )
            )
            if state_dim > state.shape[0]:
                state = torch.cat((state, torch.zeros(state_dim - state.shape[0])))
            if action_dim > action.shape[0]:
                action = torch.cat((action, torch.zeros(action_dim - action.shape[0])))
            episode_objects.append(moved)
            episode_scene.append(scene_sampled)
            episode_state.append(state)
            episode_actions.append(action)
        all_objects.append(torch.stack(episode_objects, dim=0))
        all_scene.append(torch.stack(episode_scene, dim=0))
        all_state.append(torch.stack(episode_state, dim=0))
        all_actions.append(torch.stack(episode_actions, dim=0))
    return TrajectoryTensors(
        scene_points=torch.stack(all_scene, dim=0),
        object_points=torch.stack(all_objects, dim=0),
        robot_state=torch.stack(all_state, dim=0),
        actions=torch.stack(all_actions, dim=0),
        object_mask=torch.ones(episodes, steps, objects, dtype=torch.bool),
        point_mask=torch.ones(episodes, steps, objects, object_points, dtype=torch.bool),
        episode_lengths=torch.full((episodes,), steps, dtype=torch.long),
    )

