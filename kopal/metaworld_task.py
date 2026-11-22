"""KOPAL 的 MetaWorld 环境适配器。

本文件在仿真器中使用实例级别信息生成固定对应关键物体点云，并将环境
状态和动作转换为 KOPAL 数据格式。该接口用于训练、脚本专家数据采集和
闭环评估。
"""

from __future__ import annotations

import json
import logging
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .dataset import TrajectoryTensors, save_npz_trajectory_dataset
from .geometry import farthest_point_indices, gather_points


@dataclass
class ButtonObservation:
    scene_points: torch.Tensor       # (1,Ns,3)
    object_points: torch.Tensor      # (1,1,No,3)
    robot_state: torch.Tensor        # (1,Ds)
    goal_objects: torch.Tensor       # (1,1,No,3)
    object_center: torch.Tensor      # (1,3)
    target_position: torch.Tensor    # (1,3)


def _geom_local_offsets(geom_type: int, size: np.ndarray) -> np.ndarray:
    size = np.asarray(size, dtype=np.float32)
    center = np.zeros((1, 3), dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * size[:3]
    points = [center, axes, -axes]
    if geom_type == 6:  # 盒体：沿各半轴长度加入面中心点。
        points.append((axes + np.roll(axes, 1, axis=1)) * 0.5)
    return np.concatenate(points, axis=0).astype(np.float32)


METAWORLD_ML10_TASKS = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "sweep-v3",
    "basketball-v3",
)

# 训练任务清单中额外支持的环境名称。
METAWORLD_EXP_TASKS = (
    "drawer-open-v3",
    "assembly-v3",
    "coffee-push-v3",
    "sweep-into-v3",
    "button-press-v3",
)

METAWORLD_TASKS = METAWORLD_ML10_TASKS + METAWORLD_EXP_TASKS

_OBJECT_BODY = {
    "reach-v3": "mocap",
    "push-v3": "obj",
    "pick-place-v3": "obj",
    "door-open-v3": "door_link",
    "drawer-close-v3": "drawer_link",
    "button-press-topdown-v3": "button",
    "peg-insert-side-v3": "peg",
    "window-open-v3": "windowb_a",
    "basketball-v3": "basketball",
    "drawer-open-v3": "drawer_link",
    "assembly-v3": "asmbly_peg",
    "coffee-push-v3": "mug",
    "sweep-into-v3": "obj",
    "button-press-v3": "button",
}


class MetaWorldTaskAdapter:
    """将 MetaWorld 环境封装为 KOPAL 点云轨迹接口。"""

    def __init__(
        self,
        seed: int = 0,
        scene_points: int = 256,
        object_points: int = 64,
        max_episode_steps: int = 500,
        stop_on_success: bool = True,
        task_name: str = "reach-v3",
    ) -> None:
        logging.getLogger("gymnasium").setLevel(logging.ERROR)
        try:
            import gymnasium as gym
            import metaworld  # 仅检查运行时依赖是否可导入。
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "MetaWorld dependencies are missing. Run the dependency setup "
                "with KOPAL_train/env."
            ) from error
        self.gym = gym
        self.mujoco = mujoco
        if task_name not in METAWORLD_TASKS:
            raise ValueError(
                f"Unsupported MetaWorld task {task_name!r}; available: {METAWORLD_TASKS}"
            )
        self.task_name = task_name
        policy_module = importlib.import_module("metaworld.policies")
        policy_class = policy_module.ENV_POLICY_MAP[task_name]
        self.expert_policy = policy_class()
        self.scene_point_count = scene_points
        self.object_point_count = object_points
        self.stop_on_success = stop_on_success
        self.env = gym.make(
            "Meta-World/MT1",
            env_name=self.task_name,
            max_episode_steps=max_episode_steps,
        )
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self._fixed_goal_points: Optional[torch.Tensor] = None
        self.env.reset(seed=int(seed))
        self._build_fixed_point_templates()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.shape[0])

    @property
    def state_dim(self) -> int:
        return 9

    def _build_fixed_point_templates(self) -> None:
        model = self.unwrapped.model
        object_body_name = _OBJECT_BODY.get(self.task_name)
        if object_body_name is None:
            object_geom = next(
                i for i in range(model.ngeom)
                if model.geom(i).name == "objGeom"
            )
            object_body = int(model.geom_bodyid[object_geom])
        else:
            object_body = int(model.body(object_body_name).id)
        object_geoms = [
            i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) == object_body
        ]
        if not object_geoms:
            raise RuntimeError("MetaWorld main object has no geometry")
        object_local = []
        object_geom_ids = []
        for geom_id in object_geoms:
            offsets = self._geom_local_points(geom_id, dense_mesh=True)
            object_local.append(offsets)
            object_geom_ids.extend([geom_id] * offsets.shape[0])
        object_local = np.concatenate(object_local, axis=0)
        object_tensor = torch.from_numpy(object_local)
        selected = farthest_point_indices(
            object_tensor.unsqueeze(0),
            min(self.object_point_count, object_tensor.shape[0]),
        ).squeeze(0)
        if selected.shape[0] < self.object_point_count:
            repeats = (self.object_point_count + selected.shape[0] - 1) // selected.shape[0]
            selected = selected.repeat(repeats)[: self.object_point_count]
        self.object_geom_ids = torch.tensor([object_geom_ids[int(i)] for i in selected])
        self.object_local_points = gather_points(object_tensor.unsqueeze(0), selected.unsqueeze(0)).squeeze(0)

        scene_local = []
        scene_geom_ids = []
        for geom_id in range(model.ngeom):
            offsets = self._geom_local_points(geom_id, dense_mesh=False)
            scene_local.append(offsets)
            scene_geom_ids.extend([geom_id] * offsets.shape[0])
        scene_local = np.concatenate(scene_local, axis=0)
        scene_tensor = torch.from_numpy(scene_local)
        count = min(self.scene_point_count, scene_tensor.shape[0])
        if count == self.scene_point_count:
            scene_world = self._transform_local(
                scene_tensor,
                torch.tensor(scene_geom_ids),
            )
            index = farthest_point_indices(
                scene_world.unsqueeze(0),
                count,
            ).squeeze(0)
        else:
            index = torch.linspace(0, scene_tensor.shape[0] - 1, count).long()
        if count < self.scene_point_count:
            index = torch.cat(
                (index, index[-1:].repeat(self.scene_point_count - count))
            )
        self.scene_geom_ids = torch.tensor([scene_geom_ids[int(i)] for i in index])
        self.scene_local_points = scene_tensor[index]

    def _geom_local_points(self, geom_id: int, dense_mesh: bool) -> np.ndarray:
        model = self.unwrapped.model
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(self.mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[start : start + count],
                dtype=np.float32,
            )
            if not dense_mesh and vertices.shape[0] > 12:
                index = np.linspace(0, vertices.shape[0] - 1, 12).astype(np.int64)
                vertices = vertices[index]
            return vertices
        return _geom_local_offsets(
            geom_type,
            np.asarray(model.geom_size[geom_id], dtype=np.float32),
        )

    def _transform_local(self, local_points: torch.Tensor, geom_ids: torch.Tensor) -> torch.Tensor:
        model = self.unwrapped.model
        data = self.unwrapped.data
        self.mujoco.mj_forward(model, data)
        ids = geom_ids.numpy()
        positions = torch.as_tensor(
            np.asarray(data.geom_xpos)[ids],
            dtype=local_points.dtype,
        )
        rotations = torch.as_tensor(
            np.asarray(data.geom_xmat)[ids].reshape(-1, 3, 3),
            dtype=local_points.dtype,
        )
        return torch.bmm(rotations, local_points.unsqueeze(-1)).squeeze(-1) + positions

    def reset(
        self,
        seed: Optional[int] = None,
        compute_goal: bool = False,
    ) -> ButtonObservation:
        reset_seed = self.seed if seed is None else seed
        self.env.reset(seed=int(reset_seed))
        self._fixed_goal_points = None
        if compute_goal:
            try:
                fixed_goal = self._expert_terminal_goal_points()
            except RuntimeError:
                current = self._current_object_points()
                target = torch.as_tensor(
                    np.asarray(self.unwrapped._target_pos, dtype=np.float32)
                )
                center = torch.as_tensor(
                    np.asarray(
                        self.unwrapped._get_pos_objects(),
                        dtype=np.float32,
                    )
                )
                fixed_goal = current + (target - center)
            self.env.reset(seed=int(reset_seed))
            self._fixed_goal_points = fixed_goal
        return self.observe()

    def _expert_terminal_goal_points(self) -> torch.Tensor:
        for _ in range(self.max_episode_steps):
            _, _, terminated, truncated, info = self.step(self.expert_action())
            if float(info.get("success", 0.0)) > 0.5:
                return self._current_object_points().clone()
            if terminated or truncated:
                break
        raise RuntimeError(
            f"Scripted expert failed while constructing goal for {self.task_name}"
        )

    def _current_object_points(self) -> torch.Tensor:
        return self._transform_local(
            self.object_local_points,
            self.object_geom_ids,
        )

    def shift_target(self, offset: np.ndarray) -> None:
        """平移任务目标，用于目标位置泛化评估。"""
        offset = np.asarray(offset, dtype=np.float64)
        self.unwrapped._target_pos = self.unwrapped._target_pos + offset
        try:
            self.unwrapped.model.site("goal").pos = self.unwrapped._target_pos
        except KeyError:
            pass
        if self._fixed_goal_points is not None:
            self._fixed_goal_points = self._fixed_goal_points + torch.as_tensor(
                offset,
                dtype=self._fixed_goal_points.dtype,
            )

    def translate_object(self, offset: np.ndarray) -> None:
        """在环境重置后平移主要可移动物体。"""
        offset = np.asarray(offset, dtype=np.float64)
        position = np.asarray(self.unwrapped._get_pos_objects(), dtype=np.float64)
        self.unwrapped._set_obj_xyz(position + offset)
        self.mujoco.mj_forward(
            self.unwrapped.model,
            self.unwrapped.data,
        )

    def scale_object_geometry(self, scale: float) -> None:
        """缩放主要物体的几何尺寸，用于实例泛化评估。"""
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        model = self.unwrapped.model
        geom_ids = sorted(set(int(item) for item in self.object_geom_ids))
        for geom_id in geom_ids:
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(self.mujoco.mjtGeom.mjGEOM_MESH):
                continue
            model.geom_size[geom_id] = model.geom_size[geom_id] * scale
        self._build_fixed_point_templates()

    def observe(self) -> ButtonObservation:
        model = self.unwrapped.model
        data = self.unwrapped.data
        self.mujoco.mj_forward(model, data)
        raw_obs = self.unwrapped._get_obs()
        object_points = self._current_object_points()
        scene_points = self._transform_local(self.scene_local_points, self.scene_geom_ids)
        object_center = torch.as_tensor(
            np.asarray(self.unwrapped._get_pos_objects(), dtype=np.float32)
        )
        target = torch.as_tensor(np.asarray(self.unwrapped._target_pos, dtype=np.float32))
        if self._fixed_goal_points is not None:
            goal_points = self._fixed_goal_points
        else:
            goal_shift = target - object_center
            goal_points = object_points + goal_shift
        robot_state = torch.as_tensor(
            np.asarray(
                np.concatenate(
                    (
                        self.unwrapped.get_endeff_pos(),
                        self.unwrapped._get_site_pos("rightEndEffector"),
                        self.unwrapped._get_site_pos("leftEndEffector"),
                    )
                ),
                dtype=np.float32,
            )
        )
        return ButtonObservation(
            scene_points=scene_points.unsqueeze(0),
            object_points=object_points.unsqueeze(0).unsqueeze(0),
            robot_state=robot_state.unsqueeze(0),
            goal_objects=goal_points.unsqueeze(0).unsqueeze(0),
            object_center=object_center.unsqueeze(0),
            target_position=target.unsqueeze(0),
        )

    def expert_action(self) -> np.ndarray:
        action = np.asarray(
            self.expert_policy.get_action(self.unwrapped._get_obs()),
            dtype=np.float32,
        )
        low = np.asarray(self.env.action_space.low, dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        return np.clip(action, low, high)

    def step(self, action: np.ndarray):
        return self.env.step(np.asarray(action, dtype=np.float32))

    def close(self) -> None:
        self.env.close()


MetaWorldButtonPressAdapter = MetaWorldTaskAdapter


def collect_button_press_dataset(
    output_path: str | Path,
    episodes: int = 100,
    seed: int = 0,
    scene_points: int = 256,
    object_points: int = 64,
    max_episode_steps: int = 500,
    task_name: str = "reach-v3",
) -> Dict[str, float]:
    """采集脚本专家成功轨迹并保存为本地 NPZ 数据。"""
    adapter = MetaWorldTaskAdapter(
        seed=seed,
        scene_points=scene_points,
        object_points=object_points,
        max_episode_steps=max_episode_steps,
        stop_on_success=True,
        task_name=task_name,
    )
    episode_data = []
    lengths = []
    successes = []
    try:
        attempted = 0
        while len(episode_data) < episodes:
            episode_seed = seed + attempted
            attempted += 1
            adapter.reset(episode_seed)
            scenes = []
            objects = []
            states = []
            actions = []
            success = False
            for _ in range(max_episode_steps):
                observation = adapter.observe()
                action = adapter.expert_action()
                _, _, terminated, truncated, info = adapter.step(action)
                scenes.append(observation.scene_points.squeeze(0))
                objects.append(observation.object_points.squeeze(0))
                states.append(observation.robot_state.squeeze(0))
                actions.append(torch.as_tensor(action, dtype=torch.float32))
                success = success or float(info.get("success", 0.0)) > 0.5
                if success or terminated or truncated:
                    break
            if not success:
                print(
                    f"episode_seed={episode_seed} expert_failed; retrying",
                    flush=True,
                )
                continue
            length = len(actions)
            episode_data.append(
                (
                    torch.stack(scenes),
                    torch.stack(objects),
                    torch.stack(states),
                    torch.stack(actions),
                )
            )
            lengths.append(length)
            successes.append(1.0)
            print(
                f"episode={len(episode_data):03d} seed={episode_seed} "
                f"length={length} success=1",
                flush=True,
            )
    finally:
        adapter.close()

    max_length = max(lengths)
    padded = []
    for scene, objects, state, action in episode_data:
        pad = max_length - scene.shape[0]
        if pad > 0:
            scene = torch.cat((scene, scene[-1:].expand(pad, -1, -1)), dim=0)
            objects = torch.cat((objects, objects[-1:].expand(pad, -1, -1, -1)), dim=0)
            state = torch.cat((state, state[-1:].expand(pad, -1)), dim=0)
            action = torch.cat((action, torch.zeros_like(action[-1:]).expand(pad, -1)), dim=0)
        padded.append((scene, objects, state, action))

    tensors = TrajectoryTensors(
        scene_points=torch.stack([item[0] for item in padded]),
        object_points=torch.stack([item[1] for item in padded]),
        robot_state=torch.stack([item[2] for item in padded]),
        actions=torch.stack([item[3] for item in padded]),
        object_mask=torch.ones(len(episode_data), max_length, 1, dtype=torch.bool),
        point_mask=torch.ones(
            len(episode_data),
            max_length,
            1,
            object_points,
            dtype=torch.bool,
        ),
        episode_lengths=torch.tensor(lengths, dtype=torch.long),
    )
    output_path = Path(output_path)
    save_npz_trajectory_dataset(tensors, output_path)
    metadata = {
        "task": task_name,
        "episodes": len(episode_data),
        "attempted_episodes": attempted,
        "seed": seed,
        "expert_success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "max_episode_length": int(max(lengths)),
        "padded_length": int(max_length),
        "scene_points": scene_points,
        "object_points": object_points,
        "action_dim": int(tensors.actions.shape[-1]),
        "state_dim": int(tensors.robot_state.shape[-1]),
        "dataset": str(output_path.resolve()),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata

