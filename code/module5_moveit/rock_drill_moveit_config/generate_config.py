#!/usr/bin/env python3
import os
from moveit_configs_utils import MoveItConfigsBuilder

urdf_path = "/home/wky/rock_drill_ws/src/rock_drill_description/urdf/rock_drill.urdf"
srdf_path = "/home/wky/rock_drill_ws/src/rock_drill_moveit_config/config/rock_drill.srdf"
config_dir = "/home/wky/rock_drill_ws/src/rock_drill_moveit_config/config"

# 确保配置目录存在
os.makedirs(config_dir, exist_ok=True)

# 使用 MoveItConfigsBuilder 生成配置
builder = MoveItConfigsBuilder("rock_drill", robot_description=urdf_path)
builder.robot_description_semantic(srdf_path)
builder.to_moveit_configs()

print("MoveIt configuration generated successfully in", config_dir)
