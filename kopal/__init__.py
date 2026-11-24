"""KOPAL 模型包入口。

本文件导出配置、世界模型、MeanFlow 策略、CEM 规划器、阶段标注工具和
世界模型到策略的子目标轨迹接口，供训练、推理和外部调用统一使用。
"""

from .config import KOPALConfig
from .world_model import PointFlowWorldModel
from .policy import MeanFlowPointPolicy
from .planner import ObjectTrajectoryCEMPlanner
from .interface import SubgoalTrajectoryBuffer
from .phases import (
    PhaseAnnotations,
    build_staged_trajectory,
    decode_monotonic_phases,
    infer_phase_annotations,
)

__all__ = [
    "KOPALConfig",
    "PointFlowWorldModel",
    "MeanFlowPointPolicy",
    "ObjectTrajectoryCEMPlanner",
    "SubgoalTrajectoryBuffer",
    "PhaseAnnotations",
    "infer_phase_annotations",
    "build_staged_trajectory",
    "decode_monotonic_phases",
]
