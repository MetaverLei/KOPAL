"""KOPAL 世界模型、规划器和策略的异步组合。

本文件提供统一系统对象，负责分阶段训练、策略指数滑动平均、子目标更新、
轨迹缓冲和动作块推理，是训练与闭环评估的主要入口。
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import World3DPolicyConfig
from .interface import SubgoalTrajectoryBuffer
from .planner import ObjectTrajectoryCEMPlanner, PlanningResult
from .policy import MeanFlowPointPolicy
from .world_model import ObjectPointFlowWorldModel


class WorldPolicySystem(nn.Module):
    """组合可独立训练的世界模型、CEM 规划器、轨迹缓冲器和执行策略。"""

    def __init__(self, config: Optional[World3DPolicyConfig] = None) -> None:
        super().__init__()
        self.config = config or World3DPolicyConfig()
        self.world_model = ObjectPointFlowWorldModel(self.config.world_model)
        self.policy = MeanFlowPointPolicy(self.config.policy)
        self.policy_ema = copy.deepcopy(self.policy)
        for parameter in self.policy_ema.parameters():
            parameter.requires_grad_(False)
        self.policy_ema.eval()
        self.planner = ObjectTrajectoryCEMPlanner(self.world_model, self.config.planner)
        self.subgoal_buffer = SubgoalTrajectoryBuffer(
            ema_alpha=self.config.planner.ema_alpha,
        )

    def train_world_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        self.world_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, details = self.world_model.compute_loss(
            batch["current_objects"],
            batch["target_objects"],
            actions=batch.get("world_actions"),
            object_mask=batch.get("object_mask"),
            point_mask=batch.get("point_mask"),
        )
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), grad_clip)
        optimizer.step()
        self.world_model.update_target_encoder()
        return {name: float(value.detach()) for name, value in details.items()}

    def train_policy_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        self.policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, details = self.policy.compute_loss(batch)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
        optimizer.step()
        self.update_policy_ema()
        return {name: float(value.detach()) for name, value in details.items()}

    @torch.no_grad()
    def update_policy_ema(self, decay: Optional[float] = None) -> None:
        decay = self.config.policy.ema_decay if decay is None else decay
        for target, online in zip(
            self.policy_ema.parameters(),
            self.policy.parameters(),
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)
        for target, online in zip(
            self.policy_ema.buffers(),
            self.policy.buffers(),
        ):
            target.copy_(online)

    @torch.no_grad()
    def use_policy_ema(self) -> None:
        self.policy.load_state_dict(self.policy_ema.state_dict())

    @torch.no_grad()
    def update_subgoal(
        self,
        current_objects: torch.Tensor,
        goal_objects: torch.Tensor,
        obstacle_points: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> PlanningResult:
        result = self.planner.plan(
            current_objects,
            goal_objects,
            obstacle_points=obstacle_points,
            object_mask=object_mask,
            point_mask=point_mask,
        )
        self.subgoal_buffer.update(
            result.subgoal_trajectory,
            current_points=current_objects,
            object_mask=object_mask,
            point_mask=point_mask,
        )
        return result

    @torch.no_grad()
    def act(
        self,
        scene_points: torch.Tensor,
        robot_state: torch.Tensor,
        scene_mask: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if self.subgoal_buffer.empty:
            raise RuntimeError("Run update_subgoal before querying the policy")
        subgoal = self.subgoal_buffer.get()
        output = self.policy.sample(
            scene_points,
            subgoal.points,
            robot_state,
            current_objects=subgoal.current_points,
            scene_mask=scene_mask,
            subgoal_mask=subgoal.object_mask,
            subgoal_point_mask=subgoal.point_mask,
            num_steps=num_steps,
        )
        output["subgoal_version"] = torch.tensor(
            self.subgoal_buffer.version,
            device=scene_points.device,
        )
        return output

