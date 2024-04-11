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
import time
from typing import Dict, Sequence
import matplotlib.pyplot as plt

import dill

import click
import gym
import scipy
import torch
import wandb
from gym import spaces
from hydra.core.hydra_config import HydraConfig

from analysis.predictions.robot_dataset import get_one_episode
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.real_franka_image_dataset import RandomFourierFeatures, RealFrankaImageDataset
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

from sensor_msgs.msg import Joy, JointState, CompressedImage, Image
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from franka_msgs.action import Grasp
from cv_bridge import CvBridge

import diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace

import PyKDL
from kdl_solver import KDLSolver
import copy


class EnvControlWrapperJTC:
    def __init__(self, jtc_client, n_obs_steps, n_action_steps, path_bag_robot_description):
        self.robot_description = get_robot_description(path_bag_robot_description=path_bag_robot_description)
        self.cv_bridge = CvBridge()
        self._kdl = KDLSolver(self.robot_description)
        self._kdl.set_kinematic_chain('panda_link0', 'panda_hand')

        self.observation_space = gym.spaces.Dict(
            {
                'eef': gym.spaces.Box(-8, 8, shape=(7,), dtype=np.float32),
                'camera_1': gym.spaces.Box(0, 1, shape=(3, 240, 240), dtype=np.float32),  # TODO 3 -> 4, camera_1 -> camera_0
                'mass': gym.spaces.Box( -1, 1, shape=(256,), dtype=np.float32),
                'mass_neutral': gym.spaces.Box( -1, 1, shape=(256,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32)
        self._jtc_client = jtc_client
        self._jstate = None

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = None  # TODO

        self.all_observations = collections.deque(maxlen=self.n_obs_steps)

        self.queue_actions = queue.Queue()

        # start with the initial position as a goal
        # self.initial_eef = RealFrankaImageDataset.FIXED_INITIAL_EEF
        initial_eef = np.asarray(
            [0.43996018, 0.06893278, 0.40212647, 0.0673149, 0.96574436, 0.2338243, 0.03675712]
        )  # TODO: deal with initial eef
        self.initial_eef = self.convert_eef_to_kdl(initial_eef)  # PyKDL coordinate: tr=[x,y,z] and qu = [x,y,z,w]

        self.q_prev = None  # Used for IK


    def reset(self):
        # times_joints_seq set to None, to avoid sending any action to the environment
        obs, *_ = self.step(times_joints_seq=None)  # TODO: deal with initial pos
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

    def push_actions(self, list_actions, force=False):
        raise NotImplementedError

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
        raise NotImplementedError  # not compatible with JTC

    def append_obs(self):
        obs = self.get_obs()
        self.all_observations.append(obs)

    def step(self, times_joints_seq, do_return_stacked_obs=False):
        if times_joints_seq is not None:
            self.send_jtc_action(times_joints_seq)
        obs = self.get_obs()
        reward = -1.
        info = {}
        done = False

        # self.all_observations.append(obs)

        if do_return_stacked_obs:
            # if you want the stacked obs, call request_stacked_obs() instead
            stacked_obs = self._compute_stacked_obs(n_steps=self.n_obs_steps)
        else:
            stacked_obs = None

        if (self.max_steps is not None) and (len(self.all_observations) > self.max_steps):
            done = True

        return stacked_obs, reward, done, info

    @classmethod
    def convert_eef_to_kdl(cls, eef):
        pos_x = eef[0:3]
        pos_q = eef[3:7]
        pos_q = np.asarray([pos_q[1], pos_q[2], pos_q[3], pos_q[0]])
        return np.concatenate([pos_x, pos_q])

    def convert_eef_to_joints(self,
                              eef_seq,
                              joints_init,
                              convert_eef_to_kdl_convention: bool = False,
                              verbose: bool = False):
        tq_new = []
        q_prev = None
        for i in range(0, eef_seq.shape[0]):
            # js = joint_array[i, :]
            eef = eef_seq[i]
            if convert_eef_to_kdl_convention:
                eef = self.convert_eef_to_kdl(eef)
            tr = eef[0:3]
            qu = eef[3:7]  # TODO: VERIFY THAT WE HAVE THE RIGHT CONVENTION HERE!!
            # tr, qu = self._kdl.compute_fk(js)  # PyKDL coordinate: tr=[x,y,z] and qu = [x,y,z,w]

            ee_target_pykdl = PyKDL.Frame(
                PyKDL.Rotation.Quaternion(*qu),
                PyKDL.Vector(*tr),
            )

            qpi = self._kdl.compute_ik(joints_init, ee_target_pykdl)

            if qpi is None:
                self.get_logger().error("ERROR: Fail to compute IK from the target end-effector position:")
                self.get_logger().error(
                    f"translation (xyz): {ee_target_pykdl.p}, quaternion (xyzw): {ee_target_pykdl.M.GetQuaternion()}"
                )
                exit()

            qpi = np.array(qpi)
            q_prev = qpi
            tq_new.append(qpi)

            if verbose:
                self.get_logger().info("===Original demo -> transform")
                self.get_logger().info(f"   translation (xyz)| {np.array(tr)} --> {np.array(ee_target_pykdl.p)}")
                self.get_logger().info(
                    f"   quaternion (xyzw)| {np.array(qu)} --> {np.array(ee_target_pykdl.M.GetQuaternion())}"
                )

        tq_new = np.stack(tq_new)
        return tq_new

    @classmethod
    def add_times_to_joints_seq(cls, joints_seq: Sequence[np.ndarray], frequency: int):
        period = 1. / frequency

        list_jnts_time = []
        relative_time = 0.
        for jnt in joints_seq:
            relative_time += period
            tuple_jnt_time = (relative_time, jnt)
            list_jnts_time.append(tuple_jnt_time)

        return list_jnts_time

    @classmethod
    def interpolate_joints_seq(cls, list_times_jnts, interpolate_frequency, initial_jnt, kind="cubic"):
        all_times, all_joints = list(zip(*list_times_jnts))
        all_times = np.asarray(all_times).ravel()
        all_joints = np.asarray(all_joints)

        # add initial jnt
        all_times = np.insert(all_times, 0, 0.)
        all_joints = np.vstack(tup=(initial_jnt.reshape(1, -1), all_joints))

        total_duration = all_times[-1] - all_times[0]  # = all_times[-1]

        # interpolate
        time_interpolation = np.arange(start=0.0, stop=total_duration, step=1.0 / interpolate_frequency)
        print(all_times, all_joints)
        print(len(all_times), len(all_joints))
        jnt_demo_interpolated = scipy.interpolate.interp1d(
            all_times,
            all_joints,
            axis=0,
            kind=kind,
        )(time_interpolation)

        # return the new list_times_jnts after interpolation
        return list(zip(time_interpolation, jnt_demo_interpolated))


    @classmethod
    def time_multiply(cls, list_jnts_time, time_multiplier):
        return [
            (_time * time_multiplier, joint)
            for (_time, joint) in list_jnts_time
        ]

    @classmethod
    def time_offset(cls, list_jnts_time, time_offset):
        return [
            (_time + time_offset, joint)
            for (_time, joint) in list_jnts_time
        ]

    def request_stacked_obs(self) -> Dict:
        return self._compute_stacked_obs(n_steps=self.n_obs_steps)

    def get_jstate(self):
        return self._jstate

    def set_jstate(self, msg):
        self._jstate = msg

    def jpc_send_goal(self, jpos):
        raise NotImplementedError  # not compatible with JTC

    def get_joints_pos(self):
        pos_joints = np.asarray(self._jstate.position[:7])
        return pos_joints.astype(np.float32)

    def send_jtc_action(self, jpos: list[tuple[np.ndarray, np.ndarray]]):
        """
        publish the jpos command in an appropriate message.

        Args:
            jpos (list[tuple[np.ndarray, np.ndarray]]): A list of length-2 tuple of (time, joint_position)
        """

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [f"panda_joint{i}" for i in range(1, 7+1)]

        goal_msg.trajectory.points = [
            JointTrajectoryPoint(positions=qi[:7], time_from_start=Duration(seconds=ti).to_msg()) for ti, qi in jpos
        ]
        self._jtc_client.wait_for_server()
        self._send_goal_future = self._jtc_client.send_goal_async(goal_msg)


class EnvControlWrapperWithCamerasJTC(EnvControlWrapperJTC):
    def __init__(self, jtc_client, n_obs_steps, n_action_steps, path_bag_robot_description,
                 rff_encoder: RandomFourierFeatures, mass_goal=None):
        super().__init__(jtc_client, n_obs_steps, n_action_steps, path_bag_robot_description)

        self.camera_0_compressed_msg = None
        self.camera_0_depth_compressed_msg = None

        self.mass_encoding_neutral = self._get_mass_encoding(mass=None, rff_encoder=rff_encoder)
        self.mass_encoding = self._get_mass_encoding(mass_goal, rff_encoder)

        import pathlib
        self.path_debug = pathlib.Path("images_real_debug")
        self.path_debug.mkdir(exist_ok=True)

        self._index_image = 0

    def _get_mass_encoding(self, mass, rff_encoder):
        if mass is None:
            _mass_encoding = np.array([[0., 1.]])
        else:
            _mass_encoding = np.array([[mass, 0.]])

        return rff_encoder.encode_vector(_mass_encoding)[0]

    def set_camera_0_compressed_msg(self, msg):
        self.camera_0_compressed_msg = msg

    def set_camera_0_depth_compressed_msg(self, msg):
        self.camera_0_depth_compressed_msg = msg

    def get_obs(self):
        if (self._jstate is None
                or self.camera_0_compressed_msg is None
                # or self.camera_0_depth_compressed_msg is None  # TODO
        ):
            print(f"WARNING, the following are None:")
            print(f"self._jstate: {'not None' if self._jstate is not None else 'None'}")
            print(f"self.camera_0_compressed_msg: {'not None' if self.camera_0_compressed_msg is not None else 'None'}")
            # print(
            #     f"self.camera_0_depth_compressed_msg: {'not None' if self.camera_0_depth_compressed_msg is not None else 'None'}") # TODO
            return None
        else:
            return self._compute_obs()

    def _compute_obs(self):
        pos_end_effector = end_effector_calculator(self._jstate, self._kdl)

        camera_0_data = convert_image(cv_bridge=self.cv_bridge, msg_ros=self.camera_0_compressed_msg)
        camera_0_data = self._bgr_to_rgb(camera_0_data)
        #
        # camera_0_depth_data = convert_image(cv_bridge=self.cv_bridge, msg_ros=self.camera_0_depth_compressed_msg,
        #                                     is_depth=True)
        #
        # camera_0_full_data = RealFrankaImageDataset.concatenate_rgb_depth(camera_0_data, camera_0_depth_data)  # TODO
        camera_0_full_data = camera_0_data
        camera_0_full_data = RealFrankaImageDataset.moveaxis_rgbd(camera_0_full_data, single_rgb=True)
        camera_0_full_data = RealFrankaImageDataset.rgbd_255_to_1(camera_0_full_data)

        self._save_image(camera_0_full_data.astype(np.float32))

        return {
            'eef': pos_end_effector.astype(np.float32),
            'camera_1': camera_0_full_data.astype(np.float32),  # TODO camera_1 -> camera_0
            'mass': self.mass_encoding.astype(np.float32),
            'mass_neutral': self.mass_encoding_neutral.astype(np.float32),
        }

    def _bgr_to_rgb(self, data):
        return data[..., ::-1]

    def _save_image(self, data):
        i = self._index_image

        plt.clf()
        plt.cla()
        img = data[:3]  # Taking only RGB
        print(f"Saving image {i} with shape", img.shape)
        # img = np.repeat(img, 3, axis=0)
        # print("image", i)
        img = np.moveaxis(img, 0, -1)
        plt.imshow(img)
        plt.savefig(self.path_debug / f"image_{i}.png")

        self._index_image += 1

class DiffusionController(NodeParameterMixin,
                          NodeWaitMixin,
                          NodeTFMixin,
                          rclpy.node.Node):
    NODE_PARAMETERS = dict(
        joy_topic='/spacenav/joy',
        jtc_topic="/joint_trajectory_controller/follow_joint_trajectory",
        jstate_topic='/joint_states',
        cartesian_control_topic='/cartesian_control',
        camera_0_topic='/d405rs01/color/image_rect_raw/compressed',
        camera_0_depth_topic="/d405rs01/aligned_depth_to_color/image_raw",
    )

    def __init__(self, policy, critic, n_obs_steps, n_action_steps, path_bag_robot_description, rff_encoder, mass_goal,
                 dataset,
                 *args, node_name='robot_calibrator', **kwargs):
        super().__init__(*args, node_name=node_name, node_parameters=self.NODE_PARAMETERS, **kwargs)

        self._jtc_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.jtc_topic,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
        )

        # timer
        self.timer_period = 0.10  # seconds
        self.policy_timer = self.create_timer(self.timer_period, self.policy_cb)
        self.waiting_timer = self.create_timer(0.01, self.waiting_cb)

        # robot
        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        msg = self.wait_for_message(String, '/robot_description', -1, qos_profile=latching_qos)
        self.robot_description = msg.data
        self.kdl = KDLSolver(self.robot_description)
        self.kdl.set_kinematic_chain('panda_link0', 'panda_hand')

        self.env = EnvControlWrapperWithCamerasJTC(self._jtc_client,
                                                   n_obs_steps=n_obs_steps,
                                                   n_action_steps=n_action_steps,
                                                   path_bag_robot_description=path_bag_robot_description,
                                                   rff_encoder=rff_encoder,
                                                   mass_goal=mass_goal, )
        self.policy = policy
        self.policy = self.policy.cuda()
        self.policy = self.policy.eval()

        self.policy.reset()

        self.critic = critic
        self.critic = self.critic.cuda()
        self.critic = self.critic.eval()

        # self.policy.num_inference_steps = 64

        self.stacked_obs = None

        self.start_time = None

        self.frequency = 1
        self.frequency_jstate = 10

        # joint states sub
        self.jstate_sub = self.create_subscription(
            JointState, self.jstate_topic, lambda msg: self.env.set_jstate(msg), self.frequency_jstate)

        self.camera_0_sub = self.create_subscription(
            CompressedImage, self.camera_0_topic, lambda msg: self.env.set_camera_0_compressed_msg(msg), self.frequency)

        self.camera_0_depth_sub = self.create_subscription(
            Image, self.camera_0_depth_topic, lambda msg: self.env.set_camera_0_depth_compressed_msg(msg), self.frequency)


        self._time_start_waiting = None
        self._waiting = False
        self._duration_wait = None


    def policy_cb(self):
        self.env.append_obs()
        if self.is_waiting():
            print("Waiting...")
            return

        obs = self.env.get_obs()
        if obs is None:
            print("obs is None")
            return
        jnts_obs = self.env.get_joints_pos()

        pos_x, pos_q = self.kdl.compute_fk(jnts_obs)
        # pos_q = np.asarray([pos_q[3], pos_q[0], pos_q[1], pos_q[2]])  # TODO: VERIFY ORDER CONVENTION
        eef = np.concatenate([pos_x.ravel(), pos_q.ravel()])

        delta = 5
        time_now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = time_now
        dt = (time_now - self.start_time).nanoseconds / 1e9

        # at first, just reach initial position
        if dt <= delta and delta > 0:
            print("dt <= delta and delta > 0, Initializing")
            self.stacked_obs = self.env.reset()

            jnts_target_1 = self.env.convert_eef_to_joints(self.env.initial_eef.reshape(1, -1), joints_init=jnts_obs).ravel()
            duration_init = 5  # TODO: parametrise
            times_joints_seq = [(duration_init, jnts_target_1)]
            times_joints_seq = self.env.interpolate_joints_seq(times_joints_seq, interpolate_frequency=200, initial_jnt=jnts_obs, kind="linear")
            self.env.step(times_joints_seq)
            # print(times_joints_seq)

            self.wait(duration_init + 0.05)

            # print('-' * 25)
            # print("Init part 2")
            #
            pos_x, pos_q = self.kdl.compute_fk(jnts_obs)
            # pos_q = np.asarray([pos_q[3], pos_q[0], pos_q[1], pos_q[2]])  # TODO: VERIFY ORDER CONVENTION
            eef = np.concatenate([pos_x.ravel(), pos_q.ravel()])
            print("CURRENT EEF", eef)
            #
            # tr = eef[0:3]
            # qu = eef[3:7]  # TODO: VERIFY THAT WE HAVE THE RIGHT CONVENTION HERE!!
            # # tr, qu = self._kdl.compute_fk(js)  # PyKDL coordinate: tr=[x,y,z] and qu = [x,y,z,w]
            #
            # ee_target_pykdl = PyKDL.Frame(
            #     PyKDL.Rotation.Quaternion(*qu),
            #     PyKDL.Vector(*tr),
            # )
            #
            # jnts_target_2 = self.kdl.compute_ik(jnts_obs, ee_target_pykdl)
            # duration_init = 5  # TODO: parametrise
            # times_joints_seq = [(duration_init, jnts_target_2)]
            #
            # jnts_obs = self.env.get_joints_pos()
            # times_joints_seq = self.env.interpolate_joints_seq(times_joints_seq, interpolate_frequency=200, initial_jnt=jnts_obs, kind="linear")
            # self.env.step(times_joints_seq)
            # print(times_joints_seq)
            #
            # time.sleep(duration_init + 0.05)
            return
        elif delta <= 0:
            raise NotImplementedError

        # jac


        keys_obs = ("camera_1", "eef", "mass") # TODO
        # keys_obs = ("camera_1", "eef", )  # TODO: camera_1 -> camera_0
        # keys_obs = ("eef",)

        self.get_logger().info("Adding actions to buffer")
        with torch.no_grad():
            stacked_obs = self.env.request_stacked_obs()

            stacked_neutral_obs = {
                key: value
                for key, value in stacked_obs.items()
            }
            stacked_neutral_obs["mass"] = stacked_neutral_obs["mass_neutral"]
            del stacked_obs["mass_neutral"]
            del stacked_neutral_obs["mass_neutral"]

            stacked_obs = dict_apply(stacked_obs, lambda x: x.reshape(1, *x.shape))
            stacked_neutral_obs = dict_apply(stacked_neutral_obs, lambda x: x.reshape(1, *x.shape))

            filtered_stacked_obs = dict()
            filtered_stacked_neutral_obs = dict()
            for key, value in stacked_obs.items():
                if key in keys_obs:
                    filtered_stacked_obs[key] = stacked_obs[key]
                    filtered_stacked_neutral_obs[key] = stacked_neutral_obs[key]
                else:
                    ...

            # device transfer
            obs_dict = dict_apply(filtered_stacked_obs,
                                  lambda x: torch.from_numpy(x).cuda())
            neutral_obs_dict = dict_apply(filtered_stacked_neutral_obs,
                                          lambda x: torch.from_numpy(x).cuda())

            # action_dict = self.policy.predict_action(obs_dict, neutral_obs_dict)  # TODO
            action_dict = self.policy.predict_action_from_several_samples(obs_dict, critic_network=self.critic, neutral_obs_dict=neutral_obs_dict)
            # action_dict = self.policy.predict_action(obs_dict)
            np_action_dict = dict_apply(action_dict,
                                        lambda x: x.detach().to('cpu').numpy())

            absolute_actions_eef = np_action_dict['action']

            metrics = np_action_dict['metrics']
            wandb.log(metrics)
            if absolute_actions_eef.shape[0] == 1:
                absolute_actions_eef = absolute_actions_eef.reshape(*absolute_actions_eef.shape[1:])

        print("eef", eef)
        print("absolute_actions_eef", absolute_actions_eef)
        action_joints = self.env.convert_eef_to_joints(absolute_actions_eef, joints_init=jnts_obs, convert_eef_to_kdl_convention=True)
        times_joints_seq = self.env.add_times_to_joints_seq(action_joints, frequency=5)  # TODO: frequency
        times_joints_seq = self.env.time_multiply(times_joints_seq, time_multiplier=0.5)  # TODO: parametrize
        times_joints_seq = self.env.time_offset(times_joints_seq, time_offset=0.5)  # TODO: parametrize
        times_joints_seq = self.env.interpolate_joints_seq(times_joints_seq, interpolate_frequency=200, initial_jnt=jnts_obs)  # TODO: parametrize
        duration = times_joints_seq[-1][0]
        self.env.step(times_joints_seq)
        self.wait(duration + 0.05)
        # TODO: hope observations keep getting collected in this duration.

    def is_waiting(self):
        return self._waiting

    def waiting_cb(self):
        # print("IN WAITING CB")

        # at initialization, do not wait
        if self._time_start_waiting is None:
            self._waiting = False

        if not self._waiting:
            return
        else:
            assert self._duration_wait is not None
            assert self._time_start_waiting is not None

            time_current = self.get_clock().now().nanoseconds

            if (time_current - self._time_start_waiting) / 1e9 > self._duration_wait:
                self._duration_wait = None
                self._time_start_waiting = None
                self._waiting = False
            else:
                self._waiting = True

    def wait(self, seconds):
        if self.is_waiting():
            raise RuntimeError("Already Waiting, this function should not be called twice in a row...")
        self._waiting = True
        self._duration_wait = seconds
        self._time_start_waiting = self.get_clock().now().nanoseconds


@hydra.main(
    version_base=None,
    # config_path=str(pathlib.Path(__file__).parent.joinpath(
    #     'diffusion_policy','config'))
)
def main(args=None):
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.08/14.10.05_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # only EEF
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.08/16.24.23_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/epoch=0280-mse_error_val=0.000.ckpt"  # with images, n_obs_frames_stack = 4
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.09/18.02.53_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/epoch=1530-mse_error_val=0.000.ckpt"  # with images + mass, n_obs_frames_stack = 4
    ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.10/18.53.16_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # with images + mass + critic

    # n_obs_steps = 2 # TODO
    n_obs_steps = 4
    n_action_steps = 8
    path_bag_robot_description = "/home/ros/humble/src/diffusion_policy/data/experiment_2/bags_kinesthetic_v0/rosbag_00/"

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    dataset_dir = "/home/ros/humble/src/diffusion_policy/data/fake_puree_experiments/diffusion_policy_dataset_exp2_v2/"
    cfg.task.dataset.dataset_path = dataset_dir
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # configure logging
    output_dir = HydraConfig.get().runtime.output_dir
    wandb.init(
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging
    )

    dataset = hydra.utils.instantiate(cfg.task.dataset)

    policy = workspace.model

    MASS_GOAL = 3

    args = None
    rclpy.init(args=args)
    try:
        nodes = [
            DiffusionController(policy=policy,
                                critic=workspace.critic,
                                n_obs_steps=n_obs_steps,
                                n_action_steps=n_action_steps,
                                path_bag_robot_description=path_bag_robot_description,
                                rff_encoder=dataset.rff_encoder,
                                mass_goal=MASS_GOAL,
                                dataset=dataset,
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
