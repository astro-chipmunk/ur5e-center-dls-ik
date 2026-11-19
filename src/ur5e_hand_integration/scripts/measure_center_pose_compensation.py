#!/usr/bin/env python3
"""Solve and verify 6-DOF grasp-center compensation using live TF feedback.

The center origin is the midpoint of the two virtual fingertip points. Its
orientation follows hand_base_link. This produces a separate pose compensation
table and per-angle position/orientation residuals.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_STATES_TOPIC = "/combined_joint_states"
SERVO_COMMAND_TOPIC = "/hand_servo_command"
ACTION_NAME = "/ur5e_joint_trajectory_controller/follow_joint_trajectory"
BASE_FRAME = "base_link"
CENTER_FRAME = "hand_base_link"
FINGER_LINKS = ("hand_1st_link_left_1", "hand_1st_link_right_1")
TIP_OFFSET_LOCAL = np.array((0.0, 0.0, -0.015))
DEFAULT_OUTPUT = os.path.expanduser(
    "~/ros2_ws/src/ur5e_hand_integration/scripts/center_pose_compensation_table.csv"
)


def quat_rotate(v, q):
    x, y, z = v
    qx, qy, qz, qw = q
    t = 2.0 * np.cross((qx, qy, qz), (x, y, z))
    return np.asarray((x, y, z), dtype=float) + qw * t + np.cross((qx, qy, qz), t)


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dtype=float)


def quat_inverse(q):
    return np.asarray((-q[0], -q[1], -q[2], q[3]), dtype=float) / np.dot(q, q)


def quat_rotvec(q):
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    vnorm = np.linalg.norm(q[:3])
    if vnorm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vnorm, max(-1.0, min(1.0, q[3])))
    return q[:3] * (angle / vnorm)


def pose_error(target, current):
    p_err = target[0] - current[0]
    q_err = quat_multiply(target[1], quat_inverse(current[1]))
    return np.concatenate((p_err, quat_rotvec(q_err)))


class CenterPoseCompensation(Node):
    def __init__(self, args):
        super().__init__("center_pose_compensation_solver")
        self.args = args
        self.latest_joint_state = None
        self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._joint_state_cb, 20
        )
        self.servo_pub = self.create_publisher(Float64, SERVO_COMMAND_TOPIC, 10)
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, ACTION_NAME
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def wait_until(self, predicate, timeout, label):
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.03)
            value = predicate()
            if value is not None:
                return value
        raise RuntimeError(f"Timeout waiting for {label}")

    def arm_positions(self):
        msg = self.latest_joint_state
        if msg is None:
            return None
        positions = dict(zip(msg.name, msg.position))
        if any(name not in positions for name in ARM_JOINTS):
            return None
        return np.asarray([positions[name] for name in ARM_JOINTS], dtype=float)

    def hand_position(self):
        msg = self.latest_joint_state
        if msg is None or "hand_rev13" not in msg.name:
            return None
        return float(msg.position[msg.name.index("hand_rev13")])

    def move_arm(self, positions):
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Trajectory action unavailable: {ACTION_NAME}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        duration = self.args.move_time
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        goal.trajectory.points = [point]
        sent = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
        if not sent.done() or sent.result() is None or not sent.result().accepted:
            raise RuntimeError("Trajectory goal was not accepted")
        handle = sent.result()
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=duration + 15.0
        )
        if not result_future.done() or result_future.result() is None:
            handle.cancel_goal_async()
            raise RuntimeError("Timed out waiting for trajectory result")
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"Trajectory failed: error_code={result.error_code}")

        def reached():
            actual = self.arm_positions()
            if actual is not None and np.max(np.abs(actual - positions)) < 2e-5:
                return actual
            return None
        return self.wait_until(reached, 8.0, "target UR5e joint positions")

    def set_servo(self, angle):
        if self.servo_pub.get_subscription_count() == 0:
            raise RuntimeError(f"No subscriber on {SERVO_COMMAND_TOPIC}")
        command = Float64()
        command.data = float(angle)
        end = time.monotonic() + 0.5
        while rclpy.ok() and time.monotonic() < end:
            self.servo_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.03)
        target = math.radians(float(angle) + 72.0)
        self.wait_until(
            lambda: True if self.hand_position() is not None
            and abs(self.hand_position() - target) <= 0.001 else None,
            10.0, f"hand_rev13 at servo {angle}",
        )
        self.spin_for(0.25)

    def spin_for(self, duration):
        end = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.03)

    def get_pose(self):
        # Snapshot the newest combined joint-state timestamp once. Wait until
        # robot_state_publisher has emitted TFs at least this new, so finite
        # differences never mix poses from before and after a commanded move.
        state = self.latest_joint_state
        if state is None:
            raise RuntimeError("No combined joint state available for TF sync")
        stamp = state.header.stamp
        min_stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def stamp_ns(msg):
            s = msg.header.stamp
            return int(s.sec) * 1_000_000_000 + int(s.nanosec)

        def lookup():
            try:
                transforms = [
                    self.tf_buffer.lookup_transform(
                        BASE_FRAME, link, rclpy.time.Time()
                    ) for link in FINGER_LINKS
                ]
                center_tf = self.tf_buffer.lookup_transform(
                    BASE_FRAME, CENTER_FRAME, rclpy.time.Time()
                )
                if min(stamp_ns(tf) for tf in transforms + [center_tf]) < min_stamp_ns:
                    return None
                points = []
                for tf in transforms:
                    t = tf.transform.translation
                    q = tf.transform.rotation
                    quat = np.asarray((q.x, q.y, q.z, q.w))
                    point = np.asarray((t.x, t.y, t.z)) + quat_rotate(
                        TIP_OFFSET_LOCAL, quat
                    )
                    points.append(point)
                r = center_tf.transform.rotation
                orientation = np.asarray((r.x, r.y, r.z, r.w), dtype=float)
                return ((points[0] + points[1]) / 2.0, orientation)
            except Exception:
                return None
        return self.wait_until(lookup, 10.0, "TF synchronized with joint states")

    def pose_jacobian(self, q, pose):
        jac = np.zeros((6, 6))
        eps = self.args.finite_difference
        for i in range(6):
            perturbed = q.copy()
            perturbed[i] += eps
            actual = self.move_arm(perturbed)
            shifted_pose = self.get_pose()
            jac[:3, i] = (shifted_pose[0] - pose[0]) / eps
            rot = quat_multiply(shifted_pose[1], quat_inverse(pose[1]))
            jac[3:, i] = quat_rotvec(rot) / eps
        self.move_arm(q)
        return jac

    def solve_pose(self, target, start_q, angle):
        q = start_q.copy()
        last_position = float("inf")
        last_rotation = float("inf")
        for iteration in range(self.args.max_iterations):
            q = self.move_arm(q)
            pose = self.get_pose()
            error = pose_error(target, pose)
            p_norm = float(np.linalg.norm(error[:3]))
            r_norm = float(np.linalg.norm(error[3:]))
            last_position, last_rotation = p_norm, r_norm
            if p_norm <= self.args.position_tolerance and r_norm <= self.args.rotation_tolerance:
                return q, pose, p_norm, r_norm

            jac = self.pose_jacobian(q, pose)
            # Scale angular rows so one radian is comparable to 0.1 m.
            scale = self.args.rotation_scale_m
            weighted_jac = jac.copy()
            weighted_jac[3:, :] *= scale
            weighted_error = error.copy()
            weighted_error[3:] *= scale
            lam = self.args.damping
            dq = weighted_jac.T @ np.linalg.solve(
                weighted_jac @ weighted_jac.T + (lam * lam) * np.eye(6),
                weighted_error,
            )
            max_step = float(np.max(np.abs(dq)))
            if max_step > self.args.max_joint_step:
                dq *= self.args.max_joint_step / max_step
            q = q + dq
            self.get_logger().info(
                f"servo {angle}: IK iteration {iteration + 1}, "
                f"position={p_norm * 1000:.3f} mm, rotation={math.degrees(r_norm):.4f} deg"
            )

        q = self.move_arm(q)
        pose = self.get_pose()
        error = pose_error(target, pose)
        return q, pose, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))

    def run(self):
        self.wait_until(self.arm_positions, 15.0, JOINT_STATES_TOPIC)
        if self.servo_pub.get_subscription_count() == 0:
            raise RuntimeError(f"No subscriber on {SERVO_COMMAND_TOPIC}")
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Trajectory action unavailable: {ACTION_NAME}")

        reference_q = self.arm_positions().copy()
        try:
            output = open(self.args.output, "x", newline="", encoding="utf-8")
        except FileExistsError as exc:
            raise RuntimeError(f"Output already exists: {self.args.output}") from exc

        fields = ["servo_deg", "center_x_m", "center_y_m", "center_z_m",
                  "axis_qx", "axis_qy", "axis_qz", "axis_qw",
                  "position_error_mm", "orientation_error_deg"]
        fields += [f"target_{joint}_rad" for joint in ARM_JOINTS]
        current_q = reference_q.copy()
        try:
            self.set_servo(0)
            current_q = self.move_arm(reference_q)
            reference = self.get_pose()
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            output.flush()

            for angle in self.args.angles:
                self.get_logger().info(f"Solving center pose at servo {angle} deg")
                self.set_servo(angle)
                current_q, pose, p_error, r_error = self.solve_pose(
                    reference, current_q, angle
                )
                row = {
                    "servo_deg": angle,
                    "center_x_m": pose[0][0],
                    "center_y_m": pose[0][1],
                    "center_z_m": pose[0][2],
                    "axis_qx": pose[1][0],
                    "axis_qy": pose[1][1],
                    "axis_qz": pose[1][2],
                    "axis_qw": pose[1][3],
                    "position_error_mm": p_error * 1000.0,
                    "orientation_error_deg": math.degrees(r_error),
                }
                row.update({
                    f"target_{joint}_rad": current_q[i]
                    for i, joint in enumerate(ARM_JOINTS)
                })
                writer.writerow(row)
                output.flush()
                self.get_logger().info(
                    f"servo {angle}: position error={p_error * 1000:.3f} mm, "
                    f"axis error={math.degrees(r_error):.4f} deg"
                )
        finally:
            output.close()
            try:
                self.set_servo(0)
                self.move_arm(reference_q)
            except Exception:
                pass
        self.get_logger().info(f"Pose compensation table saved: {self.args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--angles", nargs="+", type=int)
    parser.add_argument("--move-time", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--finite-difference", type=float, default=0.001)
    parser.add_argument("--position-tolerance", type=float, default=0.0005)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--rotation-scale-m", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--max-joint-step", type=float, default=0.15)
    args, ros_args = parser.parse_known_args()
    if args.angles is not None and args.all:
        parser.error("use either --angles or --all")
    args.angles = list(range(210)) if args.all else (args.angles or [0, 45, 90, 125, 150, 198, 209])
    if not args.angles or any(x < 0 or x > 209 for x in args.angles):
        parser.error("servo angles must be in 0..209")
    if len(set(args.angles)) != len(args.angles):
        parser.error("servo angles must not contain duplicates")
    args.angles.sort()
    args.rotation_tolerance = math.radians(args.rotation_tolerance_deg)
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = CenterPoseCompensation(args)
    code = 0
    try:
        node.run()
    except KeyboardInterrupt:
        code = 130
        node.get_logger().warning("Pose compensation interrupted")
    except Exception as exc:
        code = 1
        node.get_logger().error(str(exc))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
