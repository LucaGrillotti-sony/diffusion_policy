"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import collections
import queue
import os
import os.path as osp
import sys
from typing import Dict

import dill

import click
import gym
import torch
import wandb
from gym import spaces
from hydra.core.hydra_config import HydraConfig

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.real_franka_image_dataset import RandomFourierFeatures
from diffusion_policy.env_runner.real_robot_runner import RealRobot
from diffusion_policy.policy.diffusion_guided_ddim import DDIMGuidedScheduler
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import \
    TrainDiffusionTransformerLowdimWorkspace
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace
from read_sensors_utils.format_data_replay_buffer import end_effector_calculator, convert_image, get_robot_description
from read_sensors_utils.quaternion_utils import quat_library_from_ros

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

from sensor_msgs.msg import Joy, JointState, CompressedImage
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from franka_msgs.action import Grasp
from cv_bridge import CvBridge

import diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace


import PyKDL
from kdl_solver import KDLSolver
import copy


class EnvControlWrapper:
    def __init__(self, jpc_pub, n_obs_steps, n_action_steps):
        self.observation_space = gym.spaces.Dict(
            {
                'eef': gym.spaces.Box( -8, 8, shape=(7,), dtype=np.float32),
                'camera_1': gym.spaces.Box(0, 1, shape=(3, 240, 320), dtype=np.float32),
                # 'camera_2': gym.spaces.Box(0, 1, shape=(3, 240, 320), dtype=np.float32),
                # 'camera_3': gym.spaces.Box(0, 1, shape=(3, 240, 320), dtype=np.float32),
                'camera_0': gym.spaces.Box(0, 1, shape=(3, 240, 320), dtype=np.float32),
            }
        )
        # self.observation_space = gym.spaces.Box(
        #     -8, 8, shape=(7,), dtype=np.float32)  # TODO
        self.action_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32)  # TODO
        self._jpc_pub = jpc_pub
        # self.init_pos = np.load(osp.join(osp.dirname(__file__), 'init_joint_pos.npy'))
        # init_positions = np.load(osp.join(osp.dirname(__file__), 'obs_with_time.npy'))[:, 1:]
        # self.init_pos = init_positions[len(init_positions) // 2,:7]
        # self.init_pos = init_positions[len(init_positions) // 3, :7]
        # self.init_pos = np.asarray([-0.38435703, -0.82782065, 0.25952787, -2.3897604, 0.18524243, 1.5886066, 0.59382302])
        self.init_pos = np.asarray([-0.08435703, -0.62782065, 0.25952787, -2.3897604, 0.18524243, 1.5886066, 0.59382302])
        self._jstate = None

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = None  # TODO

        self.all_observations = collections.deque(maxlen=self.n_obs_steps)  # TODO: same for reward?

        self.queue_actions = queue.Queue()

    def reset(self):
        # TODO: handle reset properly
        # self.jpc_send_goal(self.init_pos)
        init_pos = self.init_pos
        obs, *_ = self.step(init_pos)
        return obs

    def _compute_obs(self):
        jnts = np.array(self._jstate.position[:7])
        np_jnts_dict = {
            'obs': jnts.astype(np.float32)
        }
        return np_jnts_dict

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
        assert (len(self.all_observations) > 0)
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

    def step(self, action, do_return_stacked_obs=False):
        self.jpc_send_goal(action)
        obs = self.get_obs()
        reward = -1.
        info = {}
        done = False

        self.all_observations.append(obs)

        if do_return_stacked_obs:
            # if you want the stacked obs, call request_stacked_obs() instead
            stacked_obs = self._compute_stacked_obs(n_steps=self.n_obs_steps)
        else:
            stacked_obs = None

        if (self.max_steps is not None) and (len(self.all_observations) > self.max_steps):
            done = True

        return stacked_obs, reward, done, info

    def request_stacked_obs(self) -> Dict:
        return self._compute_stacked_obs(n_steps=self.n_obs_steps)

    def get_jstate(self):
        return self._jstate

    def set_jstate(self, msg):
        self._jstate = msg

    def jpc_send_goal(self, jpos):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(size=7, stride=1)]
        msg.data = list(jpos)
        self._jpc_pub.publish(msg)

    def get_joints_pos(self):
        pos_joints = np.asarray(self._jstate.position[:7])
        return pos_joints.astype(np.float32)


class EnvControlWrapperWithCameras(EnvControlWrapper):
    def __init__(self, jpc_pub, n_obs_steps, n_action_steps, path_bag_robot_description, rff_encoder: RandomFourierFeatures, mass_goal=None):
        super().__init__(jpc_pub, n_obs_steps, n_action_steps)

        self.camera_1_compressed_msg = None
        # self.camera_2_compressed_msg = None
        # self.camera_3_compressed_msg = None
        self.camera_0_compressed_msg = None

        self.robot_description = get_robot_description(path_bag_robot_description=path_bag_robot_description)
        self.cv_bridge = CvBridge()
        self._kdl = KDLSolver(self.robot_description)
        self._kdl .set_kinematic_chain('panda_link0', 'panda_hand')

        self.mass_encoding = self._get_mass_encoding(mass_goal, rff_encoder)

    def _get_mass_encoding(self, mass, rff_encoder):
        if mass is None:
            _mass_encoding = np.array([[0., 1.]])
        else:
            _mass_encoding = np.array([[mass, 0.]])

        return rff_encoder.encode(_mass_encoding)[0]


    def set_camera_1_compressed_msg(self, msg):
        self.camera_1_compressed_msg = msg

    def set_camera_2_compressed_msg(self, msg):
        self.camera_2_compressed_msg = msg

    def set_camera_3_compressed_msg(self, msg):
        self.camera_3_compressed_msg = msg

    def set_camera_0_compressed_msg(self, msg):
        self.camera_0_compressed_msg = msg

    def get_obs(self):
        if (self._jstate is None
                or self.camera_1_compressed_msg is None
                # or self.camera_2_compressed_msg is None
                # or self.camera_3_compressed_msg is None
                or self.camera_0_compressed_msg is None) :
            return None
        else:
            return self._compute_obs()

    def _compute_obs(self):
        pos_end_effector = end_effector_calculator(self._jstate, self._kdl)

        camera_1_data = convert_image(cv_bridge=self.cv_bridge, msg_ros=self.camera_1_compressed_msg)
        # camera_2_data = convert_image(cv_bridge=self.cv_bridge, msg_ros=self.camera_2_compressed_msg)
        # camera_3_data = convert_image(cv_bridge=self.cv_bridge, msg_ros=self.camera_3_compressed_msg)
        camera_0_data = convert_image(cv_bridge=self.cv_bridge, msg_ros=self.camera_0_compressed_msg)

        return {
            'eef': pos_end_effector.astype(np.float32),
            'camera_1': camera_1_data.astype(np.float32),
            # 'camera_2': camera_2_data.astype(np.float32),
            # 'camera_3': camera_3_data.astype(np.float32),
            'camera_0': camera_0_data.astype(np.float32),
            'mass': self.mass_encoding.astype(np.float32)
        }


class DiffusionController(NodeParameterMixin,
                          NodeWaitMixin,
                          NodeTFMixin,
                          rclpy.node.Node):
    NODE_PARAMETERS = dict(
        joy_topic='/spacenav/joy',
        jpc_topic='/ruckig_controller/commands',
        jstate_topic='/joint_states',
        cartesian_control_topic='/cartesian_control',
        camera_0_topic='/azure06/rgb/image_raw/compressed',
        # camera_1_topic='/azure07/rgb/image_raw/compressed',
        # camera_2_topic='/azure08/rgb/image_raw/compressed',
        camera_1_topic='/d405rs01/color/image_rect_raw/compressed',
    )

    def __init__(self, policy, critic, n_obs_steps, n_action_steps, path_bag_robot_description, rff_encoder, mass_goal, *args, node_name='robot_calibrator', **kwargs):
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

        self.env = EnvControlWrapperWithCameras(self.jpc_pub,
                                                n_obs_steps=n_obs_steps,
                                                n_action_steps=n_action_steps,
                                                path_bag_robot_description=path_bag_robot_description,
                                                rff_encoder=rff_encoder,
                                                mass_goal=mass_goal,)
        self.policy = policy
        self.policy.eval().cuda()
        self.policy.reset()

        self.critic = critic
        self.critic.eval().cuda()

        # self.policy.num_inference_steps = 64

        self.stacked_obs = None

        self.start_time = None

        # joint states sub
        self.jstate_sub = self.create_subscription(
            JointState, self.jstate_topic, lambda msg: self.env.set_jstate(msg), 10)

        self.camera_0_sub = self.create_subscription(
            CompressedImage, self.camera_0_topic, lambda msg: self.env.set_camera_0_compressed_msg(msg), 10)

        # self.camera_1_sub = self.create_subscription(
        #     CompressedImage, self.camera_1_topic, lambda msg: self.env.set_camera_1_compressed_msg(msg), 10)
        #
        # self.camera_2_sub = self.create_subscription(
        #     CompressedImage, self.camera_2_topic, lambda msg: self.env.set_camera_2_compressed_msg(msg), 10)

        self.camera_1_sub = self.create_subscription(
            CompressedImage, self.camera_1_topic, lambda msg: self.env.set_camera_1_compressed_msg(msg), 10)


    def jpc_send_goal(self, jpos):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(size=7, stride=1)]
        msg.data = list(jpos)
        # self.jpc_pub.publish(msg)

    def policy_cb(self):
        obs = self.env.get_obs()
        jnts_obs = self.env.get_joints_pos()
        if obs is None:
            return

        delta = 5
        time_now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = time_now
        dt = (time_now - self.start_time).nanoseconds / 1e9
        if dt <= delta and delta > 0:
            self.stacked_obs = self.env.reset()
            return
        elif delta == 0:
            self.stacked_obs = np.asarray([self.env.get_joints_pos() for _ in range(self.env.n_action_steps)])

        if self.current_command is None:
            self.current_command = self.env.get_joints_pos()

        # joy_state = self.get_joy_state(resetp=True)
        # dx, dq = joy_state['pos']
        #
        # self.publish_cartesian_commands(dx, dq)

        # jac
        # pos_x, pos_q = self.kdl.compute_fk(jnts_obs)
        # init_pos_x, init_pos_q = self.kdl.compute_fk(self.env.init_pos)
        pos = obs["eef"]
        assert len(pos) == 7
        pos_x = pos[:3]
        pos_q = pos[3:]

        # cur_pos = se3(*self.kdl.compute_fk(jnts_obs))

        if self.env.queue_actions.empty():
            self.get_logger().info("Adding actions to buffer")
            with torch.no_grad():
                stacked_obs = self.env.request_stacked_obs()
                stacked_obs = dict_apply(stacked_obs, lambda x: x.reshape(1, *x.shape))

                # print("To", To)
                for key, value in stacked_obs.items():
                    if key == "eef":
                        continue
                    stacked_obs[key] = np.transpose(stacked_obs[key], (0, 1, 4, 2, 3))
                    print(key, stacked_obs[key].shape)

                # device transfer
                obs_dict = dict_apply(stacked_obs,
                                      lambda x: torch.from_numpy(x).cuda())

                print(dict_apply(stacked_obs, lambda x: x.shape))
                # action_dict = self.policy.predict_action(obs_dict)
                action_dict = self.policy.predict_action_from_several_samples(obs_dict, self.critic, )
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                metrics = np_action_dict['metrics']
                wandb.log(metrics)
                if action.shape[0] == 1:
                    array_dq = action.reshape(*action.shape[1:])
                else:
                    array_dq= action
                self.env.push_actions([_dq for _dq in array_dq])
                # self.env.push_actions([array_dq[0]])

        action_to_execute = self.env.get_from_queue_actions()
        action_to_execute = action_to_execute.ravel()
        # dq = action_to_execute - jnts_obs

        new_pos_x = action_to_execute[0:3]
        new_pos_q = action_to_execute[3:7].ravel()
        new_pos_q = new_pos_q / np.linalg.norm(new_pos_q)
        new_pos = se3(new_pos_x, new_pos_q)
        # new_pos_q = quat.from_float_array(new_pos_q)
        dx = (new_pos_x - pos_x)
        # dq_rot = (quat.from_float_array(pos_q).conjugate() * quat.from_float_array(init_pos_q))
        # dq_rot = (quat.from_float_array(init_pos_q) * quat.from_float_array(pos_q).conjugate())
        # print(new_pos.q, type(pos_q), type(pos_q.conjugate()))
        print(new_pos.q)
        new_pos_q = quat.from_float_array(new_pos_q)
        pos_q = quat.from_float_array(pos_q)

        dq_rot = new_pos_q * pos_q.conjugate()

        # dq_rot = quat.from_float_array([1,0,0,0])
        # self.get_logger().info(str(f"target new pos q: {new_pos_q}"))

        self.get_logger().info(str(("Predicted actions: ", dx, dq_rot, pos_q, new_pos_q)))

        J = np.array(self.kdl.compute_jacobian(jnts_obs))
        dq, *_ = np.linalg.lstsq(J, np.concatenate([dx, quat.as_rotation_vector(dq_rot)]))

        if np.max(np.abs(dq)) < 1e-4:
            return

        print(self.current_command, jnts_obs)
        self.current_command = (0.3 * self.current_command + 0.7 * jnts_obs) + dq
        # self.current_command = dq + jnts_obs

        self.get_logger().info(str(self.current_command - jnts_obs))

        self.env.step(self.current_command)


@hydra.main(
    version_base=None,
    # config_path=str(pathlib.Path(__file__).parent.joinpath(
    #     'diffusion_policy','config'))
)
def main(args=None):
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.01.16/19.33.40_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.01.26/12.15.56_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # trained to also optimize actions
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.01.31/19.22.29_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # trained to also optimize actions
    ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.02.16/17.43.43_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # trained to also optimize actions
    n_obs_steps = 2
    n_action_steps = 8
    path_bag_robot_description = "/home/ros/humble/src/read-db/rosbag2_2024_01_16-19_05_24/"

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    _config_noise_scheduler = {**copy.deepcopy(cfg.policy.noise_scheduler)}
    del _config_noise_scheduler["_target_"]
    workspace.model.noise_scheduler = DDIMGuidedScheduler(coefficient_reward=0., **_config_noise_scheduler)

    # configure logging
    output_dir = HydraConfig.get().runtime.output_dir
    wandb.init(
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging
    )

    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # workspace: BaseWorkspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    # workspace.load_checkpoint()
    #
    # workspace.model.eval()
    # workspace.model.cuda()
    # print(workspace.model.normalizer["obs"].params_dict["offset"])

    workspace.model = workspace.model.cuda()
    # workspace.model = torch.compile(workspace.model).cuda()

    args = None
    rclpy.init(args=args)
    try:
        nodes = [
            DiffusionController(policy=workspace.model,
                                critic=workspace.critic,
                                n_obs_steps=n_obs_steps,
                                n_action_steps=n_action_steps,
                                path_bag_robot_description=path_bag_robot_description,
                                rff_encoder=dataset.rff_encoder,
                                mass_goal=None,
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
