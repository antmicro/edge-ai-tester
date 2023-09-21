# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Dict, Optional, Any

from kenning.pipeline_manager.core import Node


def add_node(
        node_list: Dict[str, Node],
        nodemodule: str,
        category: str,
        type: str):
    """
    Loads a class containing Kenning block and adds it to available nodes.

    If the class can't be imported due to import errors, it is not added.

    Parameters
    ----------
    node_list : Dict[str, Node]
        List of nodes to add to the specification.
    nodemodule : str
        Python-like path to the class holding a block to add to specification.
    category : str
        Category of the block.
    type : str
        Type of the block added to the specification.
    """
    nodeclass = nodemodule.split(".")[-1]
    node_list[nodeclass] = (
        Node(nodeclass, category, type, nodemodule)
    )


def get_category_name(kenning_class):
    """
    Turns 'kenning.module.submodule1.submodule2. ... .specific_module'
    into 'module/submodule1/submodule2/...'.
    """
    names = kenning_class.__module__
    # Optionally remove kenning.
    names = re.sub(r"kenning\.", "", names)
    # Remove last class name
    names = names.split(".")[:-1]
    return '/'.join(names)


def property_value_to_dtype(value: Any) -> Optional[str]:
    if value is None:
        return None
    property_type = str(type(value).__name__)

    if property_type == 'str':
        return 'string'
    if property_type == 'int':
        return 'integer'
    if property_type == 'float':
        return 'number'
    if property_type == 'bool':
        return 'boolean'
    return None
