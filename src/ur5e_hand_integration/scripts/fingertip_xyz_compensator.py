#!/usr/bin/env python3
"""Keep fingertip XYZ fixed when only hand joints change in the display simulation."""

import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_rotation(axis, theta):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s, v = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return np.array([
        [x*x*v+c, x*y*v-z*s, x*z*v+y*s],
        [y*x*v+z*s, y*y*v+c, y*z*v-x*s],
        [z*x*v-y*s, z*y*v+x*s, z*z*v+c],
    ])


class UrdfChain:
    def __init__(self, xml_text, root_link, tip_link):
        robot = ET.fromstring(xml_text)
        joints = {}
        child_to_joint = {}
        self.limits = {}
        for element in robot.findall('joint'):
            name = element.attrib['name']
            parent = element.find('parent').attrib['link']
            child = element.find('child').attrib['link']
            origin = element.find('origin')
            xyz = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            axis_element = element.find('axis')
            axis = np.fromstring(axis_element.attrib.get('xyz', '1 0 0'), sep=' ') if axis_element is not None else np.array([1., 0., 0.])
            limit = element.find('limit')
            if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
                self.limits[name] = (float(limit.attrib['lower']), float(limit.attrib['upper']))
            joints[name] = {
                'name': name, 'parent': parent, 'child': child,
                'type': element.attrib['type'], 'xyz': xyz, 'rpy': rpy, 'axis': axis,
            }
            child_to_joint[child] = name

        chain = []
        link = tip_link
        while link != root_link:
            if link not in child_to_joint:
                raise ValueError(f'URDF has no chain from {root_link} to {tip_link}; stopped at {link}')
            joint = joints[child_to_joint[link]]
            chain.append(joint)
            link = joint['parent']
        self.chain = list(reversed(chain))

    def pose_and_jacobian(self, values):
        T = np.eye(4)
        arm_axes = {}
        for joint in self.chain:
            origin = np.eye(4)
            origin[:3, :3] = rpy_matrix(joint['rpy'])
            origin[:3, 3] = joint['xyz']
            T = T @ origin
            name = joint['name']
            kind = joint['type']
            q = float(values.get(name, 0.0))
            axis_local = joint['axis'] / max(np.linalg.norm(joint['axis']), 1e-12)
            if name in ARM_JOINTS:
                arm_axes[name] = (T[:3, :3] @ axis_local, T[:3, 3].copy())
            if kind in ('revolute', 'continuous'):
                rot = np.eye(4)
                rot[:3, :3] = axis_rotation(axis_local, q)
                T = T @ rot
            elif kind == 'prismatic':
                trans = np.eye(4)
                trans[:3, 3] = axis_local * q
                T = T @ trans

        point = T[:3, 3]
        rotation = T[:3, :3]
        J = np.zeros((6, len(ARM_JOINTS)))
        for i, name in enumerate(ARM_JOINTS):
            if name not in arm_axes:
                raise ValueError(f'URDF chain is missing arm joint {name}')
            axis_world, origin_world = arm_axes[name]
            J[:3, i] = np.cross(axis_world, point - origin_world)
            J[3:, i] = axis_world
        return point, rotation, J

    def position_and_jacobian(self, values):
        point, _, J = self.pose_and_jacobian(values)
        return point, J[:3, :]


class FingertipXyzCompensator(Node):
    def __init__(self):
        super().__init__('fingertip_xyz_compensator')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'fingertip_link')
        self.declare_parameter('damping_m_per_rad', 0.02)
        self.declare_parameter('position_tolerance_m', 1e-5)
        self.declare_parameter('max_iterations', 30)
        self.declare_parameter('max_step_rad', 0.10)
        self.declare_parameter('max_output_step_rad', 0.002)
        description = self.get_parameter('robot_description').value
        if not description:
            raise RuntimeError('robot_description parameter is empty')
        self.chain = UrdfChain(
            description,
            self.get_parameter('base_link').value,
            self.get_parameter('tip_link').value,
        )
        # Joint states are state snapshots: keep the freshest one if processing falls behind.
        self.pub = self.create_publisher(JointState, '/joint_states', 1)
        self.sub = self.create_subscription(JointState, '/desired_joint_states', self.on_desired_state, 1)
        self.previous_input = None
        self.last_output = None
        self.target_position = None
        self.compensating = False
        self.get_logger().info('Ready: hand-only changes hold fingertip XYZ; arm changes establish a new target.')

    @staticmethod
    def to_map(msg):
        return {name: float(msg.position[i]) for i, name in enumerate(msg.name) if i < len(msg.position)}

    def solve_arm(self, values, target):
        q = np.array([values.get(name, 0.0) for name in ARM_JOINTS], dtype=float)
        damping = float(self.get_parameter('damping_m_per_rad').value)
        tolerance = float(self.get_parameter('position_tolerance_m').value)
        max_iterations = int(self.get_parameter('max_iterations').value)
        max_step = float(self.get_parameter('max_step_rad').value)
        for _ in range(max_iterations):
            state = dict(values)
            state.update(dict(zip(ARM_JOINTS, q)))
            point, J = self.chain.position_and_jacobian(state)
            error = target - point
            if np.linalg.norm(error) <= tolerance:
                return q, float(np.linalg.norm(error))
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), error)
            largest = np.max(np.abs(dq))
            if largest > max_step:
                dq *= max_step / largest
            q += dq
            for i, name in enumerate(ARM_JOINTS):
                if name in self.chain.limits:
                    low, high = self.chain.limits[name]
                    q[i] = np.clip(q[i], low, high)
        state = dict(values)
        state.update(dict(zip(ARM_JOINTS, q)))
        point, _ = self.chain.position_and_jacobian(state)
        return q, float(np.linalg.norm(target - point))

    def limit_output_step(self, q):
        """Rate-limit commanded display-state changes while continuing to converge each update."""
        if self.last_output is None:
            return q
        max_step = float(self.get_parameter('max_output_step_rad').value)
        if max_step <= 0.0:
            return q
        previous = np.array([self.last_output[name] for name in ARM_JOINTS], dtype=float)
        return previous + np.clip(q - previous, -max_step, max_step)

    def on_desired_state(self, msg):
        values = self.to_map(msg)
        missing = [name for name in ARM_JOINTS if name not in values]
        if missing:
            self.get_logger().warning(f'Ignoring JointState without arm joints: {missing}')
            return
        try:
            raw_tip, _ = self.chain.position_and_jacobian(values)
        except Exception as exc:
            self.get_logger().error(f'FK failed: {exc}')
            return

        if self.previous_input is None:
            self.target_position = raw_tip.copy()
            output_values = values.copy()
        else:
            arm_changed = any(abs(values[name] - self.previous_input.get(name, values[name])) > 1e-5 for name in ARM_JOINTS)
            if arm_changed:
                # An arm command is an intentional move; remember its resulting fingertip position.
                self.target_position = raw_tip.copy()
                self.compensating = False
                output_values = values.copy()
            elif self.compensating:
                # Keep the latest compensated pose while the hand is moving or remains at its new angle.
                output_values = values.copy()
                if self.last_output is not None:
                    output_values.update({name: self.last_output[name] for name in ARM_JOINTS})
                q, error = self.solve_arm(output_values, self.target_position)
                q = self.limit_output_step(q)
                output_values.update(dict(zip(ARM_JOINTS, q)))
                if error > float(self.get_parameter('position_tolerance_m').value) * 10:
                    self.get_logger().warning(f'Fingertip target not reached; XYZ error={error * 1000:.3f} mm')
            else:
                output_values = values.copy()
                # The first hand-only change starts XYZ hold from the last displayed fingertip point.
                q, error = self.solve_arm(output_values, self.target_position)
                q = self.limit_output_step(q)
                output_values.update(dict(zip(ARM_JOINTS, q)))
                self.compensating = True
                self.get_logger().info(f'Hand changed; compensated arm. Fingertip XYZ error={error * 1000:.3f} mm')

        output = JointState()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = msg.header.frame_id
        output.name = list(msg.name)
        output.position = [output_values.get(name, float(msg.position[i])) for i, name in enumerate(msg.name)]
        output.velocity = list(msg.velocity)
        output.effort = list(msg.effort)
        self.pub.publish(output)
        self.previous_input = values
        self.last_output = output_values


def main(args=None):
    rclpy.init(args=args)
    node = FingertipXyzCompensator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
