import dataclasses
import functools
import pathlib

from pathlib import Path
from typing import List

import cv2
import numpy as np
import scipy.interpolate
from cv_bridge import CvBridge
from kdl_solver import KDLSolver
from srl_utilities.se3 import se3
import pandas as pd

from annotator.utils import interpolate
from read_sensors_utils.bag_file_parser import BagFileParser

import imagecodecs
from numcodecs.abc import Codec

import quaternion as quat

from read_sensors_utils.constants import CURRENT_EEF_POS_INTERPOLATED, TARGET_EEF_POS_INTERPOLATED


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


def convert_image(cv_bridge: CvBridge, msg_ros, is_depth=False, side_size=(200, 400)):
    if is_depth:
        img_np = cv_bridge.imgmsg_to_cv2(msg_ros)
    else:
        img_np = cv_bridge.compressed_imgmsg_to_cv2(msg_ros, "passthrough")

    if img_np is None:
        return None

    depth_min_max = [150.0, 300.0]

    if is_depth:
        # print("DEPTH", img_np.shape)
        depth = np.expand_dims(img_np, axis=-1).astype(np.float32)
        depth_nan_mask = (depth < 3.).astype(np.uint8)
        depth = np.clip(depth, a_min=depth_min_max[0], a_max=depth_min_max[1])

        kernel = np.ones((3, 3), "uint8")
        depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)
        depth[depth_nan_mask == 1] = np.abs(depth).max()
        depth_max = np.abs(depth).max()
        depth_min = np.abs(depth).min()

        depth = (depth - depth_min) / (depth_max - depth_min)
        img_np = cv2.inpaint(depth, depth_nan_mask, 1, cv2.INPAINT_NS)
        img_np = np.clip(img_np, a_min=0., a_max=1.)
        # img_np = depth
        img_np = np.asarray(255. * img_np, dtype=np.uint8)
        img_np = np.expand_dims(img_np, axis=-1)
        img_np = np.concatenate([img_np, img_np, img_np], axis=-1)

    middle_height = 100

    middle_width = img_np.shape[1] // 2 + 50
    img_np = img_np[middle_height - (side_size[0] // 2):middle_height + (side_size[0] // 2),
             middle_width - (side_size[1] // 2):middle_width + (side_size[1] // 2)]

    img_np = cv2.resize(img_np, (240, 120), interpolation=cv2.INTER_AREA)
    return img_np


def get_list_data_img(cv_bridge: CvBridge, list_images, time_offset, is_depth=False):
    list_data = []
    for index, image_t in enumerate(list_images):
        timestamp, image = image_t
        img_np = convert_image(cv_bridge=cv_bridge, msg_ros=image, is_depth=is_depth)
        if img_np is None:
            print("skipping ", index)
            continue
        list_data.append(ImageWithTimestamp(img_np, timestamp=(timestamp / 1e9) - time_offset))

    return list_data


def convert_to_arrays(list_data_img: List[ImageWithTimestamp]):
    list_img = [data.img_np for data in list_data_img]
    list_timestamps = [data.timestamp for data in list_data_img]
    return np.asarray(list_img), np.asarray(list_timestamps).ravel()


def filter_out_data(list_data, timestamps_interpolation):
    timestamps_interpolation = timestamps_interpolation.ravel()
    array_img, array_timestamps = convert_to_arrays(list_data)
    interpolator = scipy.interpolate.interp1d(array_timestamps, array_img, kind="nearest", fill_value="extrapolate",
                                              axis=0)
    res = interpolator(timestamps_interpolation)
    list_new_data = []
    for img, t in zip(res, timestamps_interpolation):
        list_new_data.append(ImageWithTimestamp(img, t.item()))
    return list_new_data


def make_video(list_images: List[ImageWithTimestamp], name, fps, is_color=True):
    import numpy as np
    import cv2
    _image = list_images[0].img_np
    if not is_color:
        _image = np.clip(_image, 0., 1.)
    size = _image.shape
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), isColor=is_color)
    for _image_data in list_images:
        img_np = _image_data.img_np
        out.write(np.uint8(img_np))
    out.release()


def collect_data_from_messages(sequence_msg, start_time, transform_fn):
    all_commands = []
    all_times_commands = []
    for index, command in enumerate(sequence_msg):
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


def end_effector_calculator(command_1, _kdl):
    if isinstance(command_1, np.ndarray):
        pos_joints = command_1[:7]
    else:
        pos_joints = command_1.position[:7]
    cur_pos = se3(*_kdl.compute_fk(np.asarray(pos_joints)))
    r = cur_pos.r
    q = quat.as_float_array(cur_pos.q)
    pos_end_effector = np.concatenate([r.ravel(), q.ravel()])
    return pos_end_effector


def treat_folder(path_load, path_save, index_episode, mass):
    path_load = Path(path_load)
    frequency = 10
    folder_name = path_load.name
    bag_file = path_load / f'{folder_name}_0.db3'

    parser = BagFileParser(bag_file)
    cv_bridge = CvBridge()
    robot_description = get_robot_description(path_load)
    kdl = KDLSolver(robot_description)
    kdl.set_kinematic_chain('panda_link0', 'panda_hand')

    print("Collecting Data")

    images_hand_eye_rgb = parser.get_messages("/d405rs01/color/image_rect_raw/compressed")
    images_hand_eye_depth = parser.get_messages("/d405rs01/aligned_depth_to_color/image_raw")

    start_time = max(images_hand_eye_rgb[0][0], images_hand_eye_depth[0][0]) / 1e9

    # Converting to numpy arrays
    data_img = dict()

    data_img["images_hand_eye_rgb"] = get_list_data_img(cv_bridge, images_hand_eye_rgb, time_offset=start_time)
    data_img["images_hand_eye_depth"] = get_list_data_img(cv_bridge,
                                                          images_hand_eye_depth,
                                                          time_offset=start_time,
                                                          is_depth=True)

    fps_dict = dict()
    fps_dict["images_hand_eye_rgb"] = 15
    fps_dict["images_hand_eye_depth"] = 15

    max_time = min(_data_list[-1].timestamp for _data_list in data_img.values())

    print("Filtering out data")
    timestamps_interpolation = np.arange(start=0, stop=max_time, step=1. / frequency)

    print("frequency", len(data_img["images_hand_eye_rgb"]) / max_time)

    for index_camera, key in enumerate(sorted(data_img)):
        # do not filter out here, the diffusion policy script does it itself
        # data_img[key] = filter_out_data(data_img[key], timestamps_interpolation)
        name_file = f"{index_camera}.mp4"
        _path_folder = path_save / "videos" / str(index_episode)
        _path_folder.mkdir(exist_ok=True, parents=True)

        make_video(data_img[key], name=str(_path_folder / name_file), fps=fps_dict[key],
                   is_color=True)  # even if it's depth, it's color

        _path_numpy = path_save / "numpy" / str(index_episode)
        _path_numpy.mkdir(exist_ok=True, parents=True)
        name_numpy_file = f"{index_camera}.npy"

        np.save(str(_path_numpy / name_numpy_file), data_img[key][0].img_np)

    print("Get End-effector")
    robot_states = parser.get_messages("/franka_robot_state_broadcaster/robot_state")
    print("robot states", len(robot_states))

    all_times_joint_states, all_end_effector_pos = collect_data_from_messages(
        robot_states,
        start_time,
        transform_fn=lambda robot_state: end_effector_calculator(command_1=np.asarray(robot_state.q), _kdl=kdl),
    )

    # Interpolating all EEF poses and THEN defining the target EEF poses
    current_eef_pos_interpolated = interpolate(all_times_joint_states, all_end_effector_pos, timestamps_interpolation)

    target_end_effector_pos_interpolated = current_eef_pos_interpolated[1:]
    current_eef_pos_interpolated = current_eef_pos_interpolated[:-1]

    print("Creating final data")
    # saving array actions
    _path_folder = path_save / "actions" / str(index_episode)
    _path_folder.mkdir(exist_ok=True, parents=True)

    np.save(file=_path_folder / TARGET_EEF_POS_INTERPOLATED,
            arr=target_end_effector_pos_interpolated)
    np.save(file=_path_folder / CURRENT_EEF_POS_INTERPOLATED,
            arr=current_eef_pos_interpolated)

    NAME_ANNOTATIONS = "annotations_video.npy"
    NAME_INTERPOLATED_ANNOTATIONS = "annotations_video_interpolated.npy"
    NAME_MASS_TXT = "mass.txt"
    _path_annotations = _path_folder / NAME_ANNOTATIONS
    _path_save_interpolated = _path_folder / NAME_INTERPOLATED_ANNOTATIONS
    _pass_mass = _path_folder / NAME_MASS_TXT
    if mass is not None:
        with open(_pass_mass, 'w') as f:
            f.write(str(mass))
    if _path_annotations.exists():
        print(f"{NAME_ANNOTATIONS} exists, generating {NAME_INTERPOLATED_ANNOTATIONS}...")
        annotations = np.load(_path_annotations)
        _, annotations_times = convert_to_arrays(data_img["images_hand_eye_rgb"])
        annotations_interpolated = interpolate(annotations_times, annotations, timestamps_interpolation)
        np.save(_path_save_interpolated, annotations_interpolated)


def read_masses_csv(path_csv):
    df = pd.read_csv(str(path_csv))
    dict_masses_per_index = dict()
    df_index = df["index"]
    df_mass = df["mass"]
    for index_row in range(len(df)):
        dict_masses_per_index[int(df_index[index_row])] = float(df_mass[index_row])
    return dict_masses_per_index


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-save", type=str, required=True)
    parser.add_argument("--path-load", type=str, required=True)
    parser.add_argument("--override", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    path_save = pathlib.Path(args.path_save).absolute()
    path_load = pathlib.Path(args.path_load).absolute()
    override = args.override

    PATH_MASSES_CSV = path_load / "data.csv"

    path_save.mkdir(exist_ok=True, parents=True)
    rosbag_paths = [file for file in path_load.iterdir() if file.name.startswith("rosbag")]

    masses_per_demo = read_masses_csv(PATH_MASSES_CSV)

    for file in path_load.iterdir():
        print(file)
    print(rosbag_paths)

    sorting_fn = lambda x: int(x.name.split("_")[1])

    for index, rosbag_paths in enumerate(sorted(rosbag_paths, key=sorting_fn)):
        print("----------------------------------------------------------------------------------------")
        print(f"Treating folder {rosbag_paths}")
        print(path_save / str(index))
        if not override and (path_save / 'numpy' / str(index)).exists():
            print(f"Skipping as {path_save / rosbag_paths.name} exists")
            continue
        treat_folder(path_load=rosbag_paths.absolute(),
                     path_save=path_save,
                     index_episode=index,
                     mass=masses_per_demo[index])


if __name__ == '__main__':
    main()
