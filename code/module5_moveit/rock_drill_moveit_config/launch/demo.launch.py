import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    urdf_path = "/home/wky/rock_drill_ws/src/rock_drill_description/urdf/rock_drill.urdf"
    srdf_path = "/home/wky/rock_drill_ws/src/rock_drill_moveit_config/config/rock_drill.srdf"
    config_dir = "/home/wky/rock_drill_ws/src/rock_drill_moveit_config/config"

    moveit_config = (
        MoveItConfigsBuilder("rock_drill")
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(file_path=srdf_path)
        .trajectory_execution(file_path=os.path.join(config_dir, "moveit_controllers.yaml"))
        .to_moveit_configs()
    )

    # robot_state_publisher：发布 TF 和 robot_description
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description]
    )

    # joint_state_publisher：配合 MoveIt 虚拟控制器发布关节状态
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"source_list": ["/move_group/fake_controller_joint_states"]}],
    )

    # MoveGroup 节点
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=[moveit_config.to_dict()],
        output="screen"
    )

    # RViz 节点
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        move_group_node,
        rviz_node,
    ])
