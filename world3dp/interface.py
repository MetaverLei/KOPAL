"""KOPAL 世界模型到策略的子目标轨迹接口。

本文件定义关键物体点云轨迹的数据结构、形状校验、异步缓冲、时间推进和
轨迹指数滑动平均。高层规划器与底层策略只通过该接口交换几何轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SubgoalTrajectory:
    points: torch.Tensor
    current_points: Optional[torch.Tensor] = None
    object_mask: Optional[torch.Tensor] = None
    point_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.points.ndim != 5 or self.points.shape[-1] != 3:
            raise ValueError(f"Subgoal points must be (B,H,O,N,3), got {tuple(self.points.shape)}")
        b, h, o, n, _ = self.points.shape
        if self.current_points is not None and self.current_points.shape != (b, o, n, 3):
            raise ValueError(
                f"current_points must be {(b, o, n, 3)}, "
                f"got {tuple(self.current_points.shape)}"
            )
        if self.object_mask is not None and self.object_mask.shape != (b, o):
            raise ValueError(f"object_mask must be {(b, o)}, got {tuple(self.object_mask.shape)}")
        if self.point_mask is not None and self.point_mask.shape != (b, o, n):
            raise ValueError(f"point_mask must be {(b, o, n)}, got {tuple(self.point_mask.shape)}")


class SubgoalTrajectoryBuffer:
    """保存规划器子目标轨迹的异步缓冲器。

    缓冲器保存世界模型输出的关键物体点云轨迹，对其执行逐点指数滑动平均，
    并支持在动作块执行后推进轨迹时间。世界模型和策略之间不进行潜变量转换。
    """

    def __init__(self, ema_alpha: float = 0.35, temporal_smoothing: float = 0.0) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= temporal_smoothing <= 1.0:
            raise ValueError("temporal_smoothing must be in [0, 1]")
        self.ema_alpha = ema_alpha
        self.temporal_smoothing = temporal_smoothing
        self._trajectory: Optional[SubgoalTrajectory] = None
        self.version = 0

    @property
    def empty(self) -> bool:
        return self._trajectory is None

    def clear(self) -> None:
        self._trajectory = None
        self.version = 0

    def update(
        self,
        points: torch.Tensor,
        current_points: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> SubgoalTrajectory:
        new_trajectory = SubgoalTrajectory(
            points.detach().clone(),
            current_points.detach().clone() if current_points is not None else None,
            object_mask.detach().clone() if object_mask is not None else None,
            point_mask.detach().clone() if point_mask is not None else None,
        )
        if self.temporal_smoothing > 0.0 and points.shape[1] >= 3:
            kernel = torch.tensor([1.0, 2.0, 1.0], device=points.device, dtype=points.dtype) / 4.0
            stacked = torch.stack(
                (
                    new_trajectory.points[:, :-2],
                    new_trajectory.points[:, 1:-1],
                    new_trajectory.points[:, 2:],
                ),
                dim=2,
            )
            smoothed = (stacked * kernel.view(1, 1, 3, 1, 1, 1)).sum(dim=2)
            new_points = new_trajectory.points.clone()
            new_points[:, 1:-1] = (
                (1.0 - self.temporal_smoothing) * new_points[:, 1:-1]
                + self.temporal_smoothing * smoothed
            )
            new_trajectory.points = new_points

        if (
            self._trajectory is not None
            and self._trajectory.points.shape == new_trajectory.points.shape
        ):
            weight = self.ema_alpha if alpha is None else alpha
            if not 0.0 < weight <= 1.0:
                raise ValueError("alpha must be in (0, 1]")
            new_trajectory.points = (
                weight * new_trajectory.points
                + (1.0 - weight) * self._trajectory.points.to(new_trajectory.points)
            )
        self.version += 1
        self._trajectory = new_trajectory
        return self.get()

    def get(self) -> SubgoalTrajectory:
        if self._trajectory is None:
            raise RuntimeError("Subgoal buffer is empty")
        return self._trajectory

    def advance(self, steps: int = 1) -> None:
        if steps < 1:
            return
        trajectory = self.get()
        steps = min(steps, trajectory.points.shape[1])
        shifted = trajectory.points[:, steps:]
        repeated = trajectory.points[:, -1:].expand(-1, steps, -1, -1, -1)
        trajectory.points = torch.cat((shifted, repeated), dim=1)
        self.version += 1

