#!/usr/bin/env python3
"""Step the hand servo and UR5e together while holding the midpoint and axes of the two fingertips."""

import csv
import math
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from hand_joint_state_publisher import calculate_hand_state

from fingertip_xyz_compensator import ARM_JOINTS, UrdfChain
from hand_joint_state_publisher import calculate_hand_state


def rotation_error(target, current):
    """Return the world-frame axis-angle error that rotates current toward target."""
    delta = target @ current.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array([
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
    ])
    if angle < 1e-8:
        return 0.5 * vee
    if abs(math.sin(angle)) < 1e-7:
        # Stable near pi: derive an axis from the largest diagonal term.
        diagonal = np.maximum((np.diag(delta) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        if axis[index] < 1e-8:
            return np.zeros(3)
        if index == 0:
            axis[1] = math.copysign(axis[1], delta[0, 1] + delta[1, 0])
            axis[2] = math.copysign(axis[2], delta[0, 2] + delta[2, 0])
        elif index == 1:
            axis[0] = math.copysign(axis[0], delta[0, 1] + delta[1, 0])
            axis[2] = math.copysign(axis[2], delta[1, 2] + delta[2, 1])
        else:
            axis[0] = math.copysign(axis[0], delta[0, 2] + delta[2, 0])
            axis[1] = math.copysign(axis[1], delta[1, 2] + delta[2, 1])
        return angle * axis / max(np.linalg.norm(axis), 1e-12)
    return angle * vee / (2.0 * math.sin(angle))


class CenterPoseMoveItCompensator(Node):
    def __init__(self):
        super().__init__("moveit_center_pose_compensator")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("left_finger_link", "hand_1st_link_left_1")
        self.declare_parameter("right_finger_link", "hand_1st_link_right_1")
        self.declare_parameter("center_frame", "hand_base_link")
        self.declare_parameter("tip_offset_m", -0.015)
        self.declare_parameter("damping_m_per_rad", 0.0005)
        self.declare_parameter("position_tolerance_m", 1e-5)
        self.declare_parameter("orientation_tolerance_rad", 0.002)
        self.declare_parameter("orientation_scale_m", 0.10)
        self.declare_parameter("tip_direction_weight", 4.0)
        self.declare_parameter("max_joint_velocity_rad_s", 0.25)
        self.declare_parameter("max_compensation_error_m", 0.00025)
        self.declare_parameter("max_compensation_error_rad", 0.05236)
        # Keep each servo sub-step close to its incoming arm configuration.
        # A hard trust region prevents wrist-seed retries from selecting a
        # distant IK branch when the local solve is near a singularity.
        self.declare_parameter("max_compensation_joint_step_rad", 0.12)
        self.declare_parameter("max_iterations", 300)
        self.declare_parameter("trajectory_batch_degrees", 1)
        self.declare_parameter("servo_interpolation_degrees", 0.25)
        self.declare_parameter("feedback_csv_path", "")
        self.declare_parameter("trajectory_action", "/ur5e_joint_trajectory_controller/follow_joint_trajectory")

        description = self.get_parameter("robot_description").value
        if not description:
            raise RuntimeError("robot_description parameter is empty")
        root = self.get_parameter("base_link").value
        self.left_chain = UrdfChain(description, root, self.get_parameter("left_finger_link").value)
        self.right_chain = UrdfChain(description, root, self.get_parameter("right_finger_link").value)
        self.axis_chain = UrdfChain(description, root, self.get_parameter("center_frame").value)
        self.limits = self.axis_chain.limits

        action_name = self.get_parameter("trajectory_action").value
        self.action = ActionClient(self, FollowJointTrajectory, action_name)
        self.state_subscription = self.create_subscription(
            JointState, "/joint_states", self.on_state, 10
        )
        self.target_subscription = self.create_subscription(
            Float64, "/hand_servo_target", self.on_servo_target, 10
        )
        self.hand_command_publisher = self.create_publisher(
            Float64, "/hand_servo_command", 10
        )

        self.previous = None
        self.latest = None
        self.held_tip = None
        self.held_rotation = None
        self.current_servo_deg = 0.0
        self.target_servo_deg = 0.0
        self.target_received = False
        self.transition_active = False
        self.goal_active = False
        self.waiting_for_hand_state = False
        self.pending_servo_deg = None
        self.batch_commands = []
        self.batch_command_index = 0
        self.batch_start_time = None
        self.batch_start_servo_deg = 0.0
        self.last_hold_feedback_time = 0.0
        self.transition_started_at = None
        self.feedback_csv_path_active = ""
        self.feedback_csv_file = None
        self.feedback_csv_writer = None
        self.batch_timer = self.create_timer(0.01, self._publish_due_hand_commands)

        self.get_logger().info(
            "Captures the current center pose and holds its position and axes during servo steps."
        )

    @staticmethod
    def _as_map(msg):
        return {
            name: float(msg.position[i])
            for i, name in enumerate(msg.name)
            if i < len(msg.position)
        }

    @staticmethod
    def _servo_from_state(values):
        if "hand_rev13" not in values:
            return 0.0
        return math.degrees(values["hand_rev13"]) - 72.0

    @staticmethod
    def _skew(v):
        x, y, z = v
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    def _tip_direction(self, values):
        directions = []
        local_tip_axis = np.array([0.0, 0.0, -1.0])
        for chain in (self.left_chain, self.right_chain):
            _, rotation, _ = chain.pose_and_jacobian(values)
            directions.append(rotation @ local_tip_axis)
        direction = np.mean(directions, axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([0.0, 0.0, -1.0])
        return direction / norm

    def _center_pose_jacobian(self, values):
        offset = np.array([0.0, 0.0, float(self.get_parameter("tip_offset_m").value)])
        point_sum = np.zeros(3)
        linear_sum = np.zeros((3, 6))
        for chain in (self.left_chain, self.right_chain):
            point, rotation, jacobian = chain.pose_and_jacobian(values)
            offset_world = rotation @ offset
            point_sum += point + offset_world
            linear_sum += jacobian[:3] - self._skew(offset_world) @ jacobian[3:]
        _, hand_rotation, hand_jacobian = self.axis_chain.pose_and_jacobian(values)
        return point_sum / 2.0, hand_rotation, np.vstack((linear_sum / 2.0, hand_jacobian[3:]))

    def _solve(self, values, target_position, target_rotation):
        initial_q = np.array([values[name] for name in ARM_JOINTS], dtype=float)
        base_damping = float(self.get_parameter("damping_m_per_rad").value)
        position_tolerance = float(self.get_parameter("position_tolerance_m").value)
        orientation_tolerance = float(self.get_parameter("orientation_tolerance_rad").value)
        base_orientation_scale = max(
            float(self.get_parameter("orientation_scale_m").value), 1e-4
        )
        tip_direction_weight = max(
            float(self.get_parameter("tip_direction_weight").value), 1.0
        )
        max_position_error = float(self.get_parameter("max_compensation_error_m").value)
        max_orientation_error = float(self.get_parameter("max_compensation_error_rad").value)
        max_joint_step = max(
            float(self.get_parameter("max_compensation_joint_step_rad").value), 1e-4
        )
        max_iterations = int(self.get_parameter("max_iterations").value)

        def evaluate(candidate):
            state = dict(values)
            state.update(dict(zip(ARM_JOINTS, candidate)))
            point, rotation, jacobian = self._center_pose_jacobian(state)
            position_error = target_position - point
            angular_error = rotation_error(target_rotation, rotation)
            tip_direction = self._tip_direction(state)
            position_weight = np.eye(3) + (tip_direction_weight - 1.0) * np.outer(
                tip_direction, tip_direction
            )
            weighted_position_error = position_weight @ position_error
            weighted_position_jacobian = position_weight @ jacobian[:3]
            weighted_jacobian = np.vstack((weighted_position_jacobian, jacobian[3:]))
            return (
                float(np.linalg.norm(position_error)),
                float(np.linalg.norm(angular_error)),
                position_error,
                angular_error,
                weighted_jacobian,
                weighted_position_error,
            )

        def optimize(seed, orientation_scale, iteration_limit):
            q = seed.copy()
            damping = base_damping
            score_best = float("inf")
            best = q.copy()
            best_position = float("inf")
            best_orientation = float("inf")
            for _ in range(iteration_limit):
                position_norm, orientation_norm, position_error, angular_error, jacobian, weighted_position_error = evaluate(q)
                score = math.hypot(
                    float(np.linalg.norm(weighted_position_error)),
                    orientation_scale * orientation_norm,
                )
                if score < score_best:
                    score_best = score
                    best = q.copy()
                    best_position = position_norm
                    best_orientation = orientation_norm
                if position_norm <= position_tolerance and orientation_norm <= orientation_tolerance:
                    return q, position_norm, orientation_norm

                weighted_j = jacobian.copy()
                weighted_j[3:] *= orientation_scale
                weighted_error = np.concatenate(
                    (weighted_position_error, orientation_scale * angular_error)
                )
                accepted = False
                trial_damping = damping
                for _ in range(8):
                    system = weighted_j @ weighted_j.T + trial_damping**2 * np.eye(6)
                    try:
                        dq = weighted_j.T @ np.linalg.solve(system, weighted_error)
                    except np.linalg.LinAlgError:
                        trial_damping *= 10.0
                        continue
                    largest = float(np.max(np.abs(dq)))
                    if largest > 0.10:
                        dq *= 0.10 / largest
                    for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                        candidate = q + fraction * dq
                        # Solve continuously around the previous accepted pose.
                        # This is a trust region per servo sub-step, not just a
                        # per-iteration clamp, so alternate seeds cannot jump branches.
                        candidate = np.clip(
                            candidate, initial_q - max_joint_step, initial_q + max_joint_step
                        )
                        for i, name in enumerate(ARM_JOINTS):
                            if name in self.limits:
                                candidate[i] = np.clip(candidate[i], *self.limits[name])
                        candidate_position, candidate_orientation, _, _, _, candidate_weighted_position = evaluate(candidate)
                        candidate_score = math.hypot(
                            float(np.linalg.norm(candidate_weighted_position)),
                            orientation_scale * candidate_orientation,
                        )
                        if candidate_score < score - max(1e-12, score * 1e-8):
                            q = candidate
                            damping = max(base_damping, trial_damping * 0.7)
                            accepted = True
                            break
                    if accepted:
                        break
                    trial_damping *= 3.0
                if not accepted:
                    break
            return best, best_position, best_orientation

        def quality(candidate):
            if float(np.max(np.abs(candidate - initial_q))) > max_joint_step + 1e-9:
                return float("inf"), float("inf")
            position_error, orientation_error, _, _, _, _ = evaluate(candidate)
            normalized_position = position_error / max(max_position_error, 1e-9)
            normalized_orientation = orientation_error / max(max_orientation_error, 1e-9)
            return max(normalized_position, normalized_orientation), (
                normalized_position**2 + normalized_orientation**2
            )

        q, position_error, orientation_error = optimize(
            initial_q, base_orientation_scale, max_iterations
        )
        candidates = [(q, position_error, orientation_error)]
        if position_error > max_position_error or orientation_error > max_orientation_error:
            # Search nearby wrist configurations and several pose trade-offs only when
            # the ordinary continuation solve cannot satisfy both error bounds.
            seeds = [q]
            for joint_index in (3, 4, 5):
                for offset in (-0.35, 0.35):
                    seed = initial_q.copy()
                    seed[joint_index] += offset
                    joint_name = ARM_JOINTS[joint_index]
                    if joint_name in self.limits:
                        seed[joint_index] = np.clip(seed[joint_index], *self.limits[joint_name])
                    seeds.append(seed)
            scales = (base_orientation_scale, base_orientation_scale * 0.5,
                      base_orientation_scale * 2.0)
            for seed in seeds:
                for scale in scales:
                    if seed is q and scale == base_orientation_scale:
                        continue
                    result = optimize(seed, scale, max(60, max_iterations))
                    candidates.append(result)
                    if result[1] <= max_position_error and result[2] <= max_orientation_error:
                        return result

        return min(candidates, key=lambda result: quality(result[0]))

    def on_state(self, msg):
        values = self._as_map(msg)
        if any(name not in values for name in ARM_JOINTS):
            return
        self.latest = values
        self.current_servo_deg = self._servo_from_state(values)
        try:
            current_tip, current_rotation, _ = self._center_pose_jacobian(values)
        except Exception as exc:
            self.get_logger().error(f"FK failed: {exc}")
            self.previous = values
            return

        if self.transition_active and self.held_tip is not None and self.held_rotation is not None:
            now = time.monotonic()
            position_delta = self.held_tip - current_tip
            position_error = float(np.linalg.norm(position_delta))
            tip_axis_error = float(np.dot(position_delta, self._tip_direction(values)))
            orientation_error = float(np.linalg.norm(rotation_error(self.held_rotation, current_rotation)))
            self._write_feedback_sample(
                values, position_delta, position_error, tip_axis_error, orientation_error
            )
            if now - self.last_hold_feedback_time >= 0.5:
                self.get_logger().info(
                    f"Servo pose feedback: center XYZ error={position_error * 1000:.3f} mm, "
                    f"tip-axis error={tip_axis_error * 1000:.3f} mm, "
                    f"axis-angle error={math.degrees(orientation_error):.3f}°."
                )
                self.last_hold_feedback_time = now

        if self.previous is None:
            self.held_tip = current_tip.copy()
            self.held_rotation = current_rotation.copy()
            self.previous = values
            if self.target_received:
                self._start_transition_if_needed()
            return

        arm_changed = any(
            abs(values[name] - self.previous.get(name, values[name])) > 1e-5
            for name in ARM_JOINTS
        )
        # The hand model has many dependent joints. Track only the servo's
        # master joint so harmless mimic-joint updates do not look like an
        # external hand command or repeatedly move the held-center reference.
        hand_changed = abs(
            values.get("hand_rev13", self.previous.get("hand_rev13", 0.0))
            - self.previous.get("hand_rev13", values.get("hand_rev13", 0.0))
        ) > 1e-5

        if not self.transition_active and not self.goal_active:
            if arm_changed:
                # MoveIt arm motion establishes a new center-pose hold target.
                self.held_tip = current_tip.copy()
                self.held_rotation = current_rotation.copy()
            elif hand_changed:
                # This can happen if another source publishes hand joints.
                prior_tip, prior_rotation, _ = self._center_pose_jacobian(self.previous)
                self.held_tip = prior_tip.copy()
                self.held_rotation = prior_rotation.copy()
                self.get_logger().warn(
                    "Hand joints changed outside the coordinated servo command."
                )

        self.previous = values

        if (
            self.transition_active
            and self.waiting_for_hand_state
            and self.pending_servo_deg is not None
            and abs(self.current_servo_deg - self.pending_servo_deg) < 0.05
        ):
            self.waiting_for_hand_state = False
            if not self.goal_active:
                self._send_next_step()

    def _write_feedback_sample(
        self, values, position_delta, position_error, tip_axis_error, orientation_error
    ):
        path = str(self.get_parameter("feedback_csv_path").value).strip()
        if not path:
            return
        if path != self.feedback_csv_path_active:
            if self.feedback_csv_file is not None:
                self.feedback_csv_file.close()
            self.feedback_csv_file = open(path, "w", newline="", encoding="utf-8")
            self.feedback_csv_writer = csv.writer(self.feedback_csv_file)
            self.feedback_csv_writer.writerow([
                "elapsed_s", "servo_deg", "target_servo_deg",
                "error_x_mm", "error_y_mm", "error_z_mm", "center_error_norm_mm",
                "tip_axis_error_mm", "axis_angle_error_deg",
            ])
            self.feedback_csv_path_active = path
        elapsed = (
            time.monotonic() - self.transition_started_at
            if self.transition_started_at is not None else 0.0
        )
        self.feedback_csv_writer.writerow([
            f"{elapsed:.6f}", f"{self.current_servo_deg:.6f}",
            f"{self.target_servo_deg:.6f}",
            f"{position_delta[0] * 1000.0:.9f}",
            f"{position_delta[1] * 1000.0:.9f}",
            f"{position_delta[2] * 1000.0:.9f}",
            f"{position_error * 1000.0:.9f}",
            f"{tip_axis_error * 1000.0:.9f}",
            f"{math.degrees(orientation_error):.9f}",
        ])
        self.feedback_csv_file.flush()

    def on_servo_target(self, msg):
        self.target_servo_deg = max(0.0, min(209.0, float(msg.data)))
        self.target_received = True
        if self.latest is not None:
            self._start_transition_if_needed()

    def _start_transition_if_needed(self):
        if self.latest is None:
            return
        if abs(self.target_servo_deg - self.current_servo_deg) < 1e-6:
            if not self.transition_active:
                return
        elif not self.transition_active:
            # Keep the original held pose across retries and successive servo targets.
            # on_state updates this reference when an intentional arm move occurs.
            if self.held_tip is None or self.held_rotation is None:
                self.held_tip, self.held_rotation, _ = self._center_pose_jacobian(self.latest)
            self.transition_active = True
            self.transition_started_at = time.monotonic()
            self.last_hold_feedback_time = 0.0
            self.get_logger().info(
                f"Coordinated servo move started: {self.current_servo_deg:.1f}° "
                f"to {self.target_servo_deg:.1f}° with interpolated center-pose hold."
            )
        self._send_next_step()

    def _publish_due_hand_commands(self):
        if self.batch_start_time is None:
            return
        elapsed = time.monotonic() - self.batch_start_time
        previous_servo = self.batch_start_servo_deg
        previous_time = 0.0
        final_servo = self.batch_start_servo_deg
        for servo_deg, point_time in self.batch_commands:
            final_servo = servo_deg
            if elapsed <= point_time:
                span = max(point_time - previous_time, 1e-6)
                fraction = max(0.0, min(1.0, (elapsed - previous_time) / span))
                command = Float64()
                command.data = previous_servo + fraction * (servo_deg - previous_servo)
                self.hand_command_publisher.publish(command)
                self.pending_servo_deg = self.batch_commands[-1][0]
                return
            previous_servo = servo_deg
            previous_time = point_time

        command = Float64()
        command.data = float(final_servo)
        self.hand_command_publisher.publish(command)
        self.pending_servo_deg = final_servo
        self.batch_start_time = None

    def _send_next_step(self):
        if (
            not self.transition_active
            or self.goal_active
            or self.waiting_for_hand_state
            or self.latest is None
            or self.held_tip is None
        ):
            return

        remaining = self.target_servo_deg - self.current_servo_deg
        if abs(remaining) < 1e-6:
            self.transition_active = False
            tip, rotation, _ = self._center_pose_jacobian(self.latest)
            position_delta = self.held_tip - tip
            position_error = float(np.linalg.norm(position_delta))
            tip_axis_error = float(np.dot(position_delta, self._tip_direction(self.latest)))
            orientation_error = float(np.linalg.norm(rotation_error(self.held_rotation, rotation)))
            self.get_logger().info(
                "MoveIt center-pose move complete; "
                f"center position error={position_error * 1000:.3f} mm, "
                f"tip-axis error={tip_axis_error * 1000:.3f} mm, "
                f"axis error={math.degrees(orientation_error):.3f}°."
            )
            return

        direction = math.copysign(1.0, remaining)
        batch_size = max(1, int(self.get_parameter("trajectory_batch_degrees").value))
        batch_degrees = min(float(batch_size), abs(remaining))
        servo_increment = max(
            float(self.get_parameter("servo_interpolation_degrees").value), 0.01
        )
        step_count = max(1, int(math.ceil(batch_degrees / servo_increment)))
        predicted_state = dict(self.latest)
        previous_q = np.array([self.latest[name] for name in ARM_JOINTS], dtype=float)
        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_JOINTS)
        batch_commands = []
        elapsed = 0.0
        max_velocity = max(
            float(self.get_parameter("max_joint_velocity_rad_s").value), 1e-3
        )
        max_position_error = float(self.get_parameter("max_compensation_error_m").value)
        max_orientation_error = float(self.get_parameter("max_compensation_error_rad").value)

        for index in range(step_count):
            step = self.current_servo_deg + direction * min(
                (index + 1.0) * servo_increment, batch_degrees
            )
            predicted_state.update(calculate_hand_state(step))
            q, position_error, orientation_error = self._solve(
                predicted_state, self.held_tip, self.held_rotation
            )
            if position_error > max_position_error or orientation_error > max_orientation_error:
                if not trajectory.points:
                    self.transition_active = False
                    axial_error = float(np.dot(
                        self.held_tip - self._center_pose_jacobian(predicted_state)[0],
                        self._tip_direction(predicted_state),
                    ))
                    self.get_logger().error(
                        f"Servo stopped at {self.current_servo_deg:.0f}°: "
                        f"center error={position_error * 1000:.3f} mm "
                        f"(tip-axis component={axial_error * 1000:.3f} mm), "
                        f"axis error={math.degrees(orientation_error):.3f}° "
                        f"(limits {max_position_error * 1000:.2f} mm and "
                        f"{math.degrees(max_orientation_error):.2f}°; "
                        f"max joint step {math.degrees(float(self.get_parameter('max_compensation_joint_step_rad').value)):.1f}°). "
                        "No collision checker is connected to this node; check the trajectory controller / planning-scene logs."
                    )
                    return
                break

            delta = q - previous_q
            duration = max(0.05, float(np.max(np.abs(delta))) / max_velocity)
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            elapsed += duration
            point.time_from_start.sec = int(elapsed)
            point.time_from_start.nanosec = int((elapsed - int(elapsed)) * 1e9)
            trajectory.points.append(point)
            batch_commands.append((step, elapsed))
            predicted_state.update(dict(zip(ARM_JOINTS, q)))
            previous_q = q

        if not trajectory.points:
            return

        self.batch_commands = batch_commands
        self.batch_command_index = 0
        self.batch_start_servo_deg = self.current_servo_deg
        self.batch_start_time = None
        self.pending_servo_deg = batch_commands[-1][0]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance.sec = 2
        self.goal_active = True
        self.waiting_for_hand_state = True
        future = self.action.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._abort_transition(f"Could not send compensation trajectory: {exc}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self._abort_transition("Trajectory controller rejected the compensation goal.")
            return

        self.batch_start_time = time.monotonic()
        self._publish_due_hand_commands()
        goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        self.goal_active = False
        self.batch_commands = []
        self.batch_command_index = 0
        self.batch_start_time = None
        try:
            wrapped = future.result()
            result = wrapped.result
            if (
                wrapped.status != GoalStatus.STATUS_SUCCEEDED
                or result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
            ):
                self._abort_transition(
                    f"Compensation trajectory failed (status={wrapped.status}, "
                    f"error_code={result.error_code}). "
                    "Inspect controller logs for path/goal tolerance or collision-stop details; "
                    "the FollowJointTrajectory result does not identify collision pairs."
                )
                return
        except Exception as exc:
            self._abort_transition(f"Could not read compensation result: {exc}")
            return

        if (
            self.pending_servo_deg is not None
            and abs(self.current_servo_deg - self.pending_servo_deg) < 0.05
        ):
            self.waiting_for_hand_state = False
            self._send_next_step()
        else:
            self.waiting_for_hand_state = True

    def _abort_transition(self, message):
        self.goal_active = False
        self.waiting_for_hand_state = False
        self.transition_active = False
        self.pending_servo_deg = None
        self.batch_commands = []
        self.batch_command_index = 0
        self.batch_start_time = None
        self.get_logger().error(message)


def main(args=None):
    rclpy.init(args=args)
    node = CenterPoseMoveItCompensator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
