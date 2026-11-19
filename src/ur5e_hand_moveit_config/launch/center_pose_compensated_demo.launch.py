from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_share = Path(get_package_share_directory('ur5e_hand_moveit_config'))
    moveit_config = (
        MoveItConfigsBuilder('ur5e_with_hand', package_name='ur5e_hand_moveit_config')
        .planning_pipelines(default_planning_pipeline='ompl', pipelines=['ompl'])
        .trajectory_execution(
            file_path='config/moveit_controllers.yaml',
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
    )

    ld = LaunchDescription()
    move_group_configuration = {
        'publish_robot_description_semantic': True,
        'allow_trajectory_execution': True,
        'capabilities': moveit_config.move_group_capabilities['capabilities'],
        'disable_capabilities': moveit_config.move_group_capabilities['disable_capabilities'],
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'monitor_dynamics': False,
        # The Python fake controller publishes/interpolates states at a modest
        # rate; give the desktop simulation enough time to settle before
        # MoveIt reports CONTROL_FAILED for an otherwise completed motion.
        'trajectory_execution.allowed_execution_duration_scaling': 3.0,
        'trajectory_execution.allowed_goal_duration_margin': 2.0,
    }
    ld.add_action(Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        parameters=[moveit_config.to_dict(), move_group_configuration],
        remappings=[('/joint_states', '/move_group_joint_states')],
        output='screen',
    ))
    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', str(moveit_share / 'config' / 'moveit.rviz')],
        parameters=[
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        output='screen',
    ))

    # One state publisher consumes the combined arm + hand state.
    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[moveit_config.robot_description],
        output='screen',
    ))

    # Keep an independent visual snapshot of the planned start pose.
    ld.add_action(Node(
        package='ur5e_hand_integration',
        executable='start_state_latcher.py',
        name='start_state_latcher',
        output='screen',
    ))
    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='start_state_robot_state_publisher',
        parameters=[moveit_config.robot_description, {'frame_prefix': 'start_state/'}],
        remappings=[('/joint_states', '/start_state_joint_states')],
        output='screen',
    ))
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='start_state_world_transform',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--roll', '0', '--pitch', '0',
                   '--yaw', '0', '--frame-id', 'world', '--child-frame-id', 'start_state/world'],
        output='screen',
    ))

    # Fake action server drives the arm state for a standalone MoveIt/RViz simulation.
    ld.add_action(Node(
        package='ur5e_hand_integration',
        executable='moveit_fake_arm_controller.py',
        name='moveit_fake_arm_controller',
        output='screen',
    ))

    ld.add_action(Node(
        package='ur5e_hand_integration',
        executable='hand_joint_state_publisher.py',
        name='hand_joint_state_publisher',
        output='screen',
    ))
    # Combine raw arm state and the hand model, then publish the single state stream
    # used by MoveIt and robot_state_publisher.
    ld.add_action(Node(
        package='ur5e_hand_integration',
        executable='joint_state_combiner.py',
        name='joint_state_combiner',
        remappings=[
            ('/joint_states', '/raw_arm_joint_states'),
            ('/combined_joint_states', '/joint_states'),
        ],
        output='screen',
    ))
    ld.add_action(Node(
        package='ur5e_hand_integration',
        executable='moveit_center_hand_compensator.py',
        name='moveit_center_pose_compensator',
        parameters=[
            moveit_config.robot_description,
            {'base_link': 'base_link'},
            {'left_finger_link': 'hand_1st_link_left_1'},
            {'right_finger_link': 'hand_1st_link_right_1'},
            {'center_frame': 'hand_base_link'},
            {'tip_offset_m': -0.015},
            {'damping_m_per_rad': 0.0005},
            {'max_iterations': 300},
            {'position_tolerance_m': 0.0005},
            {'orientation_tolerance_rad': 0.005},
            {'max_compensation_joint_step_rad': 0.12},
            {'orientation_scale_m': 0.10},
            {'tip_direction_weight': 1.0},
            {'max_compensation_error_m': 0.0005},
            {'max_compensation_error_rad': 0.005},
            {'max_joint_velocity_rad_s': 0.36},
            {'trajectory_batch_degrees': 1},
            {'servo_interpolation_degrees': 0.25},
            {'trajectory_action': '/ur5e_joint_trajectory_controller/follow_joint_trajectory'},
        ],
        output='screen',
    ))

    return ld
