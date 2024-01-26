import dataclasses
import functools
import pathlib
import pickle

import subprocess
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import rclpy
import scipy.interpolate
from cv_bridge import CvBridge
from numcodecs import register_codec
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from kdl_solver import KDLSolver
from srl_utilities.se3 import SE3, se3, se3_mat, se3_mul, se3_repr, se3_unmat, _rw2wr, lie_grad



from read_sensors_utils.bag_file_parser import BagFileParser

import matplotlib.pyplot as plt


import imagecodecs
from numcodecs.abc import Codec

import quaternion as quat


CURRENT_EEF_POS_INTERPOLATED = "current_eef_pos_interpolated.npy"
TARGET_EEF_POS_INTERPOLATED = "target_end_effector_pos_interpolated.npy"


def protective_squeeze(x: np.ndarray):
    """
    Squeeze dim only if it's not the last dim.
    Image dim expected to be *, H, W, C
    """
    img_shape = x.shape[-3:]
    if len(x.shape) > 3:
        n_imgs = np.prod(x.shape[:-3])
        if n_imgs > 1:
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)


class Jpeg2k(Codec):
    """JPEG 2000 codec for numcodecs."""

    codec_id = 'imagecodecs_jpeg2k'

    def __init__(
        self,
        level=None,
        codecformat=None,
        colorspace=None,
        tile=None,
        reversible=None,
        bitspersample=None,
        resolutions=None,
        numthreads=None,
        verbose=0,
    ):
        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.tile = None if tile is None else tuple(tile)
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.verbose = verbose

    def encode(self, buf):
        buf = protective_squeeze(np.asarray(buf))
        return imagecodecs.jpeg2k_encode(
            buf,
            level=self.level,
            codecformat=self.codecformat,
            colorspace=self.colorspace,
            tile=self.tile,
            reversible=self.reversible,
            bitspersample=self.bitspersample,
            resolutions=self.resolutions,
            numthreads=self.numthreads,
            verbose=self.verbose,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpeg2k_decode(
            buf, verbose=self.verbose, numthreads=self.numthreads, out=out
        )

@dataclasses.dataclass
class ImageWithTimestamp:
    img_np: np.ndarray
    timestamp: float


def convert_image(cv_bridge: CvBridge, msg_ros):
    msg_fmt = "passthrough"
    img_np = cv_bridge.compressed_imgmsg_to_cv2(msg_ros, msg_fmt)
    img_np = cv2.resize(img_np, (320, 240), interpolation=cv2.INTER_AREA)
    return img_np


def get_list_data_img(cv_bridge: CvBridge, list_images, time_offset):
    list_data = []
    for image_t in list_images:
        timestamp, image = image_t
        img_np = convert_image(cv_bridge=cv_bridge, msg_ros=image)
        list_data.append(ImageWithTimestamp(img_np, timestamp=(timestamp / 1e9) - time_offset))
        # print((timestamp / 1e9) - time_offset)
    return list_data


def convert_to_arrays(list_data_img: List[ImageWithTimestamp]):
    list_img = [data.img_np for data in list_data_img]
    list_timestamps = [data.timestamp for data in list_data_img]
    return np.asarray(list_img), np.asarray(list_timestamps).ravel()


def filter_out_data(list_data, timestamps_interpolation):
    timestamps_interpolation = timestamps_interpolation.ravel()
    array_img, array_timestamps = convert_to_arrays(list_data)
    interpolator = scipy.interpolate.interp1d(array_timestamps, array_img, kind="nearest", fill_value="extrapolate", axis=0)
    res = interpolator(timestamps_interpolation)
    list_new_data = []
    for img, t in zip(res, timestamps_interpolation):
        list_new_data.append(ImageWithTimestamp(img, t.item()))
    return list_new_data

def make_video(list_images: List[ImageWithTimestamp], name, fps):
    import numpy as np
    import cv2
    _image = list_images[0].img_np
    size = _image.shape
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
    for _image_data in list_images:
        out.write(np.uint8(_image_data.img_np))
    out.release()


def collect_data_from_messages(sequence_msg, start_time, transform_fn):
    all_commands = []
    all_times_commands = []
    for command in sequence_msg:
        time_cartesian_commands = command[0] / 1e9
        all_commands.append(transform_fn(command[1]))
        all_times_commands.append(time_cartesian_commands)
    all_commands = np.asarray(all_commands)
    all_times_commands = np.asarray(all_times_commands) - start_time
    return all_times_commands, all_commands


def get_robot_description(path_bag_robot_description):
    path_load = Path(path_bag_robot_description)
    folder_name = path_load.name
    bag_file = path_load / f'{folder_name}_0.db3'
    parser = BagFileParser(bag_file)
    return parser.get_messages("/robot_description")[-1][1].data


def interpolate(x, y, new_x):
    interpolator = scipy.interpolate.interp1d(x, y, axis=0,
                                              kind="nearest", fill_value="extrapolate", )
    target_end_effector_pos_interpolated = interpolator(new_x)
    return target_end_effector_pos_interpolated


def end_effector_calculator(command_1, _kdl):
    pos_joints = command_1.position[:7]
    cur_pos = se3(*_kdl.compute_fk(np.asarray(pos_joints)))
    r = cur_pos.r
    q = quat.as_float_array(cur_pos.q)
    pos_end_effector = np.concatenate([r.ravel(), q.ravel()])
    return pos_end_effector

def treat_folder(path_load, path_save, index_episode):
    path_load = Path(path_load)
    frequency = 10
    folder_name = path_load.name
    bag_file = path_load / f'{folder_name}_0.db3'
    
    parser = BagFileParser(bag_file)
    # images_front_camera = parser.get_messages("/d405rs01/color/image_rect_raw/compressed")
    cv_bridge = CvBridge()
    # print(images_azure_06)

    PATH_BAG_ROBOT_DESCRIPTION = "rosbag2_2024_01_11-17_13_10/" # TODO

    robot_description = get_robot_description(PATH_BAG_ROBOT_DESCRIPTION)
    kdl = KDLSolver(robot_description)
    kdl.set_kinematic_chain('panda_link0', 'panda_hand')

    print("Collecting Data")
    images_azure_06 = parser.get_messages("/azure06/rgb/image_raw/compressed")
    # images_azure_07 = parser.get_messages("/azure07/rgb/image_raw/compressed")
    # images_azure_08 = parser.get_messages("/azure08/rgb/image_raw/compressed")
    images_hand_eye = parser.get_messages("/d405rs01/color/image_rect_raw/compressed")

    # start_time = max(images_azure_06[0][0], images_azure_07[0][0], images_azure_08[0][0], images_hand_eye[0][0]) / 1e9
    start_time = max(images_azure_06[0][0], images_hand_eye[0][0]) / 1e9

    # Converting to numpy arrays
    print("Converting Data to Numpy Arrays and reshape")
    data_img = dict()
    data_img["azure_06"] = get_list_data_img(cv_bridge, images_azure_06, time_offset=start_time)
    # data_img["azure_07"] = get_list_data_img(cv_bridge, images_azure_07, time_offset=start_time)
    # data_img["azure_08"] = get_list_data_img(cv_bridge, images_azure_08, time_offset=start_time)
    data_img["images_hand_eye"] = get_list_data_img(cv_bridge, images_hand_eye, time_offset=start_time)

    fps_dict = dict()
    fps_dict["azure_06"] = 30
    # fps_dict["azure_07"] = 30
    # fps_dict["azure_08"] = 30
    fps_dict["images_hand_eye"] = 15

    max_time = min(_data_list[-1].timestamp for _data_list in data_img.values())

    print(start_time, max_time)

    print("Filtering out data")
    timestamps_interpolation = np.arange(start=0, stop=max_time, step=1./frequency)

    print("frequency", len(data_img["azure_06"]) / max_time)

    for index_camera, key in enumerate(sorted(data_img)):
        # do not filter out here, the diffusion policy script does it itself
        # data_img[key] = filter_out_data(data_img[key], timestamps_interpolation)
        name_file = f"{index_camera}.mp4"
        _path_folder = path_save / "videos" / str(index_episode)
        _path_folder.mkdir(exist_ok=True, parents=True)
        make_video(data_img[key], name=str(_path_folder / name_file), fps=fps_dict[key])

    print("Get End-effector")
    target_end_effector_poses = parser.get_messages("/cartesian_control")
    joint_states = parser.get_messages("/joint_states")

    all_times_cartesian_commands, all_end_effector_targets = collect_data_from_messages(target_end_effector_poses, start_time, transform_fn=lambda x: np.asarray(x.data))
    all_times_joint_states, all_end_effector_pos = collect_data_from_messages(joint_states, start_time, transform_fn=functools.partial(end_effector_calculator, _kdl=kdl))  # TODO

    target_end_effector_pos_interpolated = interpolate(all_times_cartesian_commands, all_end_effector_targets, timestamps_interpolation)
    current_eef_pos_interpolated = interpolate(all_times_joint_states, all_end_effector_pos, timestamps_interpolation)

    print("Creating final data")

    for key, value in data_img.items():
        array_img, array_timestamps = convert_to_arrays(value)
        data_img[key] = array_img

    # # testing compression video.
    # import zarr
    #
    # register_codec(Jpeg2k)
    #
    # this_compressor = Jpeg2k(level=50)
    # print("Shapes", data_img["azure_06"].shape, (1, *data_img["azure_06"].shape[1:]), target_end_effector_pos_interpolated.shape)
    # z = zarr.array(data_img["azure_06"], chunks=(1, *data_img["azure_06"].shape[1:]), compressor=this_compressor, dtype=np.uint8)
    # print(z.info)

    # saving array actions
    _path_folder = path_save / "actions" / str(index_episode)
    _path_folder.mkdir(exist_ok=True, parents=True)
    np.save(file=_path_folder / TARGET_EEF_POS_INTERPOLATED,
            arr=target_end_effector_pos_interpolated)
    np.save(file=_path_folder / CURRENT_EEF_POS_INTERPOLATED,
            arr=current_eef_pos_interpolated)


def main():
    PATH_TO_LOAD = pathlib.Path("saved_rosbags").absolute()
    PATH_SAVE = pathlib.Path("data_output").absolute()
    PATH_SAVE.mkdir(exist_ok=True, parents=True)
    rosbag_paths = [file for file in PATH_TO_LOAD.iterdir() if file.name.startswith("rosbag")]
    for index, rosbag_paths in enumerate(sorted(rosbag_paths)):
        print(f"Treating folder {rosbag_paths.name}")
        treat_folder(path_load=rosbag_paths.absolute(),
                     path_save=PATH_SAVE,
                     index_episode=index)

if __name__ == '__main__':
    main()
