"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from isaacsim.core.utils.prims import get_prim_at_path  # type: ignore
from isaacsim.core.utils.semantics import add_labels  # type: ignore


def set_semantic_label(prim_path: str, label: str) -> None:
    prim = get_prim_at_path(prim_path)
    add_labels(prim, [label], instance_name="class")
    # prim = get_prim_at_path(prim_path)
    # if prim.GetTypeName() == "Mesh":
    #     add_update_semantics(prim, semantic_label=label, type_label="class")
    # all_children = prim.GetAllChildren()
    # for child in all_children:
    #     set_semantic_label(str(child.GetPath()), label)
