"""
Bring up the perception costmap node.

    ros2 launch perception_costmap perception.launch.py
    ros2 launch perception_costmap perception.launch.py image_topic:=/carla/rgb \
        lidar_topic:=/carla/lidar rviz:=true

Paths are resolved from the installed package share dir -- no hardcoded
home directories. Sensor topics are launch args so the same launch works for
CARLA and for the real car (just point the args at the right topics).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('perception_costmap')
    default_cfg = os.path.join(pkg, 'config', 'perception_costmap.yaml')

    cfg = LaunchConfiguration('config')
    image_topic = LaunchConfiguration('image_topic')
    lidar_topic = LaunchConfiguration('lidar_topic')
    use_rviz = LaunchConfiguration('rviz')

    args = [
        DeclareLaunchArgument('config', default_value=default_cfg,
                              description='perception_costmap params YAML'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image'),
        DeclareLaunchArgument('lidar_topic', default_value='/lidar/points'),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]

    perception = Node(
        package='perception_costmap',
        executable='costmap_node',
        name='perception_costmap',
        output='screen',
        parameters=[cfg],
        remappings=[
            ('/camera/image', image_topic),
            ('/lidar/points', lidar_topic),
        ],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen', condition=IfCondition(use_rviz),
    )

    return LaunchDescription(args + [perception, rviz])
