"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field


@dataclass
class PointCloudInfo:
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    scale: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
    trans: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))


@dataclass
class MeshInfo:
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
    trans: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    scale: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))
