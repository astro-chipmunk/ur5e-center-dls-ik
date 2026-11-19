# UR5e Center-Pose Damped Least-Squares IK

## Scope

The custom solver uses a URDF-derived center-pose Jacobian and damped least-squares updates. The companion measurement node uses finite-difference Jacobians for the six arm joints.

`moveit_center_hand_compensator.py` is the online center-pose compensator. `measure_center_pose_compensation.py` records the independent sweep data used to study the solver. `fingertip_xyz_compensator.py` provides shared URDF-chain kinematics.

This repository isolates the selected method from the larger TRiSPAN ROS 2 workspace. It preserves the original ROS package names and the model/configuration files needed to understand the experiment. ROS 2 Humble, MoveIt 2, the Universal Robots description package, and the remaining workspace runtime dependencies are external prerequisites. Use the complete workspace archive for the original launch environment.

## Source layout

- `src/ur5e_hand_integration/scripts/`: solver and required mock/runtime helper nodes.
- `src/ur5e_hand_integration/urdf/` and `meshes/`: the UR5e hand model assets used by the experiment.
- `src/ur5e_hand_moveit_config/config/`: the MoveIt SRDF and solver/controller configuration.
- `src/ur5e_hand_moveit_config/launch/center_pose_compensated_demo.launch.py`: the method-specific RViz mock launch.

## Reproduction context

Launch `ur5e_hand_moveit_config center_pose_compensated_demo.launch.py` in the archived ROS 2 workspace. The archived dataset and run logs are kept separately from this source repository.

The archived RViz mock comparison reports 210/210 servo-angle samples within 0.5 mm position and 0.1 degree orientation tolerance. The separate finite-difference sweep used a 0.001 rad perturbation, 0.5 mm position tolerance, 0.1 degree orientation tolerance, damping 0.01, and at most 8 iterations. These results are simulation-only.

The reported values describe the recorded mock experiment and are not a claim of real-robot accuracy. No license was declared in the original source package, so this repository does not add a license grant.
