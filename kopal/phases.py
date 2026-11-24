"""KOPAL 操作轨迹的阶段标注、轨迹分段与单调阶段解码工具。

本文件从夹爪事件和关键物体运动自动推断接近、抓取和运输阶段，并将阶段
标签压缩为固定对应点云轨迹，使世界模型到策略的接口仍保持为
``(B,H,O,N,3)`` 的几何轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Optional

import torch


@dataclass
class PhaseAnnotations:
    """一条或多条轨迹的阶段标签、边界、目标点、掩码和阶段数。"""

    phase_ids: torch.Tensor
    phase_boundary: torch.Tensor
    phase_goal_points: torch.Tensor
    phase_mask: torch.Tensor
    phase_count: torch.Tensor


def infer_phase_annotations(
    object_points: torch.Tensor,
    actions: torch.Tensor,
    episode_lengths: Optional[torch.Tensor],
    *,
    num_phases: int = 3,
    gripper_channel: int = 3,
) -> PhaseAnnotations:
    """根据夹爪事件和关键物体运动推断接近、抓取和运输阶段。

    第一次夹爪闭合作为抓取边界；随后物体质心首次产生显著位移的位置作为
    运输边界。没有夹爪事件的成功轨迹退化为单阶段。
    """
    episodes, steps, objects, points, _ = object_points.shape
    phase_ids = torch.zeros(episodes, steps, objects, dtype=torch.long)
    phase_boundary = torch.zeros(episodes, steps, objects, dtype=torch.bool)
    phase_goal_points = torch.zeros(
        episodes, num_phases, objects, points, 3, dtype=object_points.dtype
    )
    phase_mask = torch.zeros(episodes, num_phases, dtype=torch.bool)
    phase_count = torch.full((episodes,), num_phases, dtype=torch.long)

    lengths = (
        episode_lengths
        if episode_lengths is not None
        else torch.full((episodes,), steps, dtype=torch.long)
    )
    for episode in range(episodes):
        length = max(int(lengths[episode]), 3)
        gripper = actions[episode, :length, gripper_channel]
        closed = gripper > 0.5
        transitions = torch.nonzero(
            closed[1:] != closed[:-1],
            as_tuple=False,
        ).flatten()
        closing_transitions = torch.nonzero(
            closed[1:] & ~closed[:-1],
            as_tuple=False,
        ).flatten()
        if num_phases == 3 and transitions.numel() == 0:
            phase_ids[episode, :length] = 0
            phase_boundary[episode, length - 1] = True
            phase_goal_points[episode, 0] = object_points[episode, length - 1]
            phase_mask[episode, 0] = True
            phase_count[episode] = 1
            continue
        if num_phases != 3:
            boundaries = [
                max(1, length * phase // num_phases)
                for phase in range(1, num_phases)
            ]
            close_index = boundaries[0] if boundaries else length
            transport_index = boundaries[1] if len(boundaries) > 1 else length
        else:
            if closing_transitions.numel() > 0:
                close_index = int(closing_transitions[0]) + 1
            else:
                close_index = max(1, length // 3)
            centres = object_points[episode, :length, :, :, :].mean(dim=-2)
            displacement = torch.linalg.vector_norm(
                centres - centres[close_index - 1 : close_index],
                dim=-1,
            ).amax(dim=-1)
            total_displacement = torch.linalg.vector_norm(
                centres[-1] - centres[0],
                dim=-1,
            )
            threshold = torch.clamp(total_displacement * 0.1, min=0.01)
            moving = displacement > threshold
            candidates = torch.nonzero(
                moving[close_index:].any(dim=-1),
                as_tuple=False,
            ).flatten()
            transport_index = (
                int(candidates[0]) + close_index
                if candidates.numel() > 0
                else min(length, close_index + max(2, length // 10))
            )
            transport_index = min(max(transport_index, close_index), length)

        phase_ids[episode, :close_index] = 0
        phase_ids[episode, close_index:transport_index] = 1
        phase_ids[episode, transport_index:length] = 2
        used_phases = min(num_phases, 3)
        for phase, boundary in enumerate(
            (close_index - 1, transport_index - 1, length - 1)
        ):
            if phase >= used_phases:
                break
            phase_boundary[episode, boundary] = True
            phase_goal_points[episode, phase] = object_points[episode, boundary]
        phase_mask[episode, :used_phases] = True
        phase_count[episode] = used_phases

    return PhaseAnnotations(
        phase_ids,
        phase_boundary,
        phase_goal_points,
        phase_mask,
        phase_count,
    )


def segment_trajectory_by_phase(
    trajectory: torch.Tensor,
    phase_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回阶段目标点、阶段边界和每个样本的阶段数。

    ``trajectory`` 的形状为 ``(B,H,O,N,3)``，``phase_ids`` 的形状为
    ``(B,H,O)``。
    """
    if trajectory.ndim != 5 or phase_ids.shape != trajectory.shape[:3]:
        raise ValueError("trajectory and phase_ids have incompatible shapes")
    phase_ids = torch.cummax(phase_ids, dim=1).values
    boundaries = torch.zeros_like(phase_ids, dtype=torch.bool)
    if phase_ids.shape[1] > 1:
        boundaries[:, :-1] = phase_ids[:, 1:] != phase_ids[:, :-1]
    boundaries[:, -1] = True
    batch, steps, objects = phase_ids.shape
    max_phases = int(phase_ids.max().item()) + 1
    goals = trajectory.new_zeros(batch, max_phases, objects, *trajectory.shape[3:])
    for batch_index in range(batch):
        for phase in range(max_phases):
            object_goals = []
            for object_index in range(objects):
                indices = torch.nonzero(
                    (phase_ids[batch_index, :, object_index] == phase)
                    & boundaries[batch_index, :, object_index],
                    as_tuple=False,
                ).flatten()
                index = int(indices[-1]) if indices.numel() else steps - 1
                object_goals.append(trajectory[batch_index, index, object_index])
            goals[batch_index, phase] = torch.stack(object_goals)
    counts = phase_ids.amax(dim=(1, 2)) + 1
    return goals, boundaries, counts


def build_staged_trajectory(
    trajectory: torch.Tensor,
    phase_ids: torch.Tensor,
    target_steps: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将带阶段标签的 rollout 压缩为一条阶段化点云轨迹。

    返回轨迹保持 ``(B,H,O,N,3)`` 接口，并保证每个阶段端点都出现在序列中。
    单阶段轨迹保持原始点流；多阶段轨迹在相邻阶段端点之间插值。
    """
    if trajectory.ndim != 5 or phase_ids.shape != trajectory.shape[:3]:
        raise ValueError("trajectory and phase_ids have incompatible shapes")
    phase_ids = torch.cummax(phase_ids.long(), dim=1).values
    batch, steps, objects = phase_ids.shape
    output_steps = steps if target_steps is None else int(target_steps)
    if output_steps < 1:
        raise ValueError("target_steps must be positive")
    max_phases = int(phase_ids.max().item()) + 1
    goals = trajectory.new_zeros(
        batch,
        max_phases,
        objects,
        *trajectory.shape[3:],
    )
    source_boundaries = torch.zeros_like(phase_ids, dtype=torch.bool)
    if steps > 1:
        source_boundaries[:, :-1] = phase_ids[:, 1:] != phase_ids[:, :-1]
    source_boundaries[:, -1] = True
    staged = trajectory.new_zeros(
        batch,
        output_steps,
        objects,
        *trajectory.shape[3:],
    )

    for batch_index in range(batch):
        for object_index in range(objects):
            object_phase = phase_ids[batch_index, :, object_index]
            counts = torch.bincount(object_phase, minlength=max_phases)
            present = [phase for phase in range(max_phases) if counts[phase] > 0]
            endpoints = []
            for phase in present:
                indices = torch.nonzero(
                    (object_phase == phase) & source_boundaries[batch_index, :, object_index],
                    as_tuple=False,
                ).flatten()
                endpoint = int(indices[-1])
                endpoints.append(endpoint)
                goals[batch_index, phase, object_index] = trajectory[
                    batch_index, endpoint, object_index
                ]

            cumulative = torch.cumsum(counts[present].double(), dim=0)
            allocated_ends = torch.round(
                cumulative / cumulative[-1] * output_steps
            ).long()
            allocated_ends = torch.clamp(
                allocated_ends, min=1, max=output_steps
            )
            for index in range(1, len(allocated_ends)):
                allocated_ends[index] = torch.maximum(
                    allocated_ends[index],
                    allocated_ends[index - 1] + 1,
                )
            allocated_ends[-1] = output_steps
            start_index = 0
            start_points = trajectory[batch_index, 0, object_index]
            for phase, end_index in zip(present, allocated_ends.tolist()):
                goal = goals[batch_index, phase, object_index]
                end_offset = end_index - 1
                if len(present) == 1 and output_steps == steps:
                    staged[batch_index, :, object_index] = trajectory[
                        batch_index, :, object_index
                    ]
                    start_index = end_index
                    start_points = goal
                    continue
                segment_length = end_offset - start_index + 1
                fractions = torch.linspace(
                    0.0,
                    1.0,
                    segment_length,
                    dtype=trajectory.dtype,
                    device=trajectory.device,
                )
                staged[batch_index, start_index:end_index, object_index] = (
                    start_points.unsqueeze(0)
                    + fractions.unsqueeze(-1).unsqueeze(-1)
                    * (goal - start_points).unsqueeze(0)
                )
                start_index = end_index
                start_points = goal

    counts = phase_ids.amax(dim=(1, 2)) + 1
    return staged, goals, source_boundaries, counts


def decode_monotonic_phases(
    phase_logits: torch.Tensor,
    phase_count_logits: torch.Tensor,
    max_phases: Optional[int] = None,
) -> torch.Tensor:
    """解码预测的连续单调阶段序列。

    ``phase_logits`` 的形状为 ``(B,S,H,O,C)``，``phase_count_logits`` 的
    形状为 ``(B,S,C)``。阶段数预测头输出终止阶段编号；解码器从第一个
    rollout 状态预测的当前阶段开始，因此中途重规划不会重置到接近阶段。
    ``max_phases=1`` 会将结果强制为单阶段。
    """
    if phase_logits.ndim != 5:
        raise ValueError("phase_logits must be (B,S,H,O,C)")
    batch, samples, steps, objects, classes = phase_logits.shape
    log_prob = torch.log_softmax(phase_logits, dim=-1)
    counts = phase_count_logits.argmax(dim=-1).clamp(0, classes - 1) + 1
    decoded = phase_logits.new_zeros(
        batch,
        samples,
        steps,
        objects,
        dtype=torch.long,
    )
    cumulative = torch.cumsum(log_prob, dim=2)
    for batch_index in range(batch):
        for sample_index in range(samples):
            for object_index in range(objects):
                start_phase = int(
                    phase_logits[
                        batch_index, sample_index, 0, object_index
                    ].argmax()
                )
                end_phase = max(
                    start_phase,
                    int(counts[batch_index, sample_index]) - 1,
                )
                if max_phases is not None:
                    end_phase = min(end_phase, int(max_phases) - 1)
                    start_phase = min(start_phase, end_phase)
                count = end_phase - start_phase + 1
                if count == 1 or steps == 1:
                    decoded[
                        batch_index,
                        sample_index,
                        :,
                        object_index,
                    ] = start_phase
                    continue
                partitions = list(
                    itertools.combinations(range(1, steps), count - 1)
                )
                scores = []
                for partition in partitions:
                    boundaries = (0, *partition, steps)
                    score = phase_logits.new_zeros(())
                    for offset, phase in enumerate(
                        range(start_phase, end_phase + 1)
                    ):
                        start, end = boundaries[offset], boundaries[offset + 1]
                        segment = (
                            cumulative[
                                batch_index,
                                sample_index,
                                end - 1,
                                object_index,
                                phase,
                            ]
                            - (
                                cumulative[
                                    batch_index,
                                    sample_index,
                                    start - 1,
                                    object_index,
                                    phase,
                                ]
                                if start > 0
                                else 0.0
                            )
                        )
                        score = score + segment
                    scores.append(score)
                partition = partitions[int(torch.stack(scores).argmax())]
                boundaries = (0, *partition, steps)
                for offset, phase in enumerate(
                    range(start_phase, end_phase + 1)
                ):
                    decoded[
                        batch_index,
                        sample_index,
                        boundaries[offset] : boundaries[offset + 1],
                        object_index,
                    ] = phase
    return decoded
