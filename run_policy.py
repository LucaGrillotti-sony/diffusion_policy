"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import queue
import os
import os.path as osp
import sys

import click
import gym
import torch
from gym import spaces

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.real_robot_runner import RealRobot
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import \
    TrainDiffusionTransformerLowdimWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import numpy as np
import numpy.linalg as la
import quaternion as quat
import rclpy
import rclpy.node
import rclpy.qos
from rclpy.duration import Duration

from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from srl_utilities.node2 import NodeParameterMixin, NodeTFMixin, NodeWaitMixin
from srl_utilities.se3 import SE3, se3, se3_mat, se3_mul, se3_repr, se3_unmat, _rw2wr, lie_grad

from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from franka_msgs.action import Grasp

import PyKDL
from kdl_solver import KDLSolver
import copy

class EnvControlWrapper:
    def __init__(self, jpc_pub, n_obs_steps, n_action_steps):
        self.observation_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32)  # TODO
        self.action_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32)  # TODO
        self._jpc_pub = jpc_pub
        self.init_pos = np.load(osp.join(osp.dirname(__file__), 'init_joint_pos.npy'))
        self._jstate = None

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = None  # TODO

        self.all_observations = []  # TODO: same for reward?

        self.queue_actions = queue.Queue()

    def reset(self):
        # TODO: handle reset properly
        #self.jpc_send_goal(self.init_pos)
        init_pos = self.init_pos
        obs, *_ = self.step(init_pos)
        return obs

    def _compute_obs(self):
        jnts = np.array(self._jstate.position[:7])
        return jnts

    def get_obs(self):
        if self._jstate is None:
            return None
        else:
            return self._compute_obs()

    def push_actions(self, list_actions):
        if not self.queue_actions.empty():
            raise ValueError("Queue actions is not empty, cannot push anything new to it")
        assert len(list_actions) == self.n_action_steps
        for action in list_actions:
            self.queue_actions.put(action)

    def _compute_stacked_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.all_observations) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return self.stack_last_n_obs(self.all_observations, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = self.stack_last_n_obs(
                    [obs[key] for obs in self.all_observations],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    @classmethod
    def stack_last_n_obs(cls, all_obs, n_steps):
        all_obs = list(all_obs)
        result = np.zeros((n_steps,) + all_obs[-1].shape,
                          dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
        return result

    def get_from_queue_actions(self):
        if self.queue_actions.empty():
            raise ValueError("Queue actions should not be empty when calling step")

        action = self.queue_actions.get()
        return action

    def step(self, action):
        self.jpc_send_goal(action)
        obs = self.get_obs()
        reward = -1.
        info = {}
        done = False

        self.all_observations.append(obs)
        stacked_obs = self._compute_stacked_obs(n_steps=self.n_obs_steps)

        if (self.max_steps is not None) and (len(self.all_observations) > self.max_steps):
            done = True

        return stacked_obs, reward, done, info  # TODO

    def get_jstate(self):
        return self._jstate

    def set_jstate(self, msg):
        self._jstate = msg

    def jpc_send_goal(self, jpos):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(size=7, stride=1)]
        msg.data = list(jpos)
        print("goal sent", list(jpos))
        self._jpc_pub.publish(msg)

class DiffusionController(NodeParameterMixin,
                          NodeWaitMixin,
                          NodeTFMixin,
                          rclpy.node.Node):
    NODE_PARAMETERS = dict(
        joy_topic='/spacenav/joy',
        # jpc_topic=('jpc_topic', '/joint_group_position_controller/commands'),
        jpc_topic='/ruckig_controller/commands',
        # jpc_topic=('jpc_topic', '/joint_group_velocity_controller/commands'),
        jstate_topic='/joint_states',
    )

    def __init__(self, policy, n_obs_steps, n_action_steps, *args, node_name='robot_calibrator', **kwargs):
        super().__init__(*args, node_name=node_name, node_parameters=self.NODE_PARAMETERS, **kwargs)
        # jtc commandor
        self.current_command = None
        self.jpc_pub = self.create_publisher(Float64MultiArray, self.jpc_topic, 10)

        # timer
        self.timer_period = 0.10  # seconds
        self.policy_timer = self.create_timer(self.timer_period, self.policy_cb)

        # robot
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        msg = self.wait_for_message(String, '/robot_description', -1, qos_profile=latching_qos)
        self.robot_description = msg.data
        self.kdl = KDLSolver(self.robot_description)
        self.kdl.set_kinematic_chain('panda_link0', 'panda_hand')

        self.env = EnvControlWrapper(self.jpc_pub, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)  # TODO: magic constants.
        self.policy = policy
        self.policy.eval().cuda()
        self.policy.reset()

        self.stacked_obs = None

        self.reset_counter = 20
        self.start_time = None

        # joint states sub
        self.jstate_sub = self.create_subscription(
            JointState, self.jstate_topic, lambda msg: self.env.set_jstate(msg), 10)

    def jpc_send_goal(self, jpos):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(size=7, stride=1)]
        msg.data = list(jpos)
        self.jpc_pub.publish(msg)

    def policy_cb(self):
        jnts_obs = self.env.get_obs()
        if jnts_obs is None:
            return

        delta = 10
        time_now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = time_now
        dt = (time_now - self.start_time).nanoseconds / 1e9
        if dt <= delta:
            self.stacked_obs = self.env.reset()
            return

        if self.current_command is None:
            self.current_command = jnts_obs

        # joy_state = self.get_joy_state(resetp=True)
        # dx, dq = joy_state['pos']
        #
        # self.publish_cartesian_commands(dx, dq)

        # jac
        # cur_pos = se3(*self.kdl.compute_fk(jnts_obs))


        if self.env.queue_actions.empty():
            with torch.no_grad():
                np_obs_dict = {
                    'obs': self.stacked_obs.astype(np.float32)
                }

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).cuda())
                action_dict = self.policy.predict_action(obs_dict)
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                array_dq = action.reshape(*action.shape[1:])

                self.env.push_actions([_dq for _dq in array_dq])
                # self.env.push_actions([array_dq[0]])

        # self.publish_dq(dq)

        action_to_execute = self.env.get_from_queue_actions()
        action_to_execute = action_to_execute.ravel()
        dx = action_to_execute[0:3]
        dq_rot = quat.from_float_array(action_to_execute[3:7])

        J = np.array(self.kdl.compute_jacobian(jnts_obs))
        dq, *_ = np.linalg.lstsq(J, np.concatenate([dx, quat.as_rotation_vector(dq_rot)]))

        if np.max(np.abs(dq)) < 1e-2:
            return

        self.current_command = (0.3 * self.current_command + 0.7 * jnts_obs) + dq

        self.get_logger().info(str(self.current_command - jnts_obs))

        self.stacked_obs, *_ = self.env.step(self.current_command)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg, args=None):
    OmegaConf.resolve(cfg)

    workspace: BaseWorkspace = TrainDiffusionTransformerLowdimWorkspace(cfg)
    workspace.load_checkpoint()
    workspace.model.eval()
    workspace.model.cuda()
    print(workspace.model.normalizer["obs"].params_dict["offset"])

    workspace.model = torch.compile(workspace.model).cuda()

    rclpy.init(args=args)
    try:
        nodes = [
            DiffusionController(policy=workspace.model,
                                n_obs_steps=4,
                                n_action_steps=8,
                                ),
        ]

        executor = rclpy.executors.MultiThreadedExecutor(4)
        for ni in nodes:
            executor.add_node(ni)
        try:
            executor.spin()
        finally:
            executor.shutdown()
        for ni in nodes:
            ni.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

