from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib

import cv2
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures

from threadpoolctl import threadpool_limits
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)

register_codecs()


def real_data_to_replay_buffer(
    dataset_path: str,
    out_store: Optional[zarr.ABSStore] = None,
    out_resolutions: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
    lowdim_keys: Optional[Sequence[str]] = None,
    image_keys: Optional[Sequence[str]] = None,
    lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
    image_compressor: Optional[numcodecs.abc.Codec] = None,
    n_decoding_threads: int = min(multiprocessing.cpu_count(), 16),
    n_encoding_threads: int = min(multiprocessing.cpu_count(), 16),
    max_inflight_tasks: int = min(multiprocessing.cpu_count(), 16) * 5,
    verify_read: bool = True,
    dt: int = None,
) -> ReplayBuffer:
    """
    It is recommended to use before calling this function
    to avoid CPU oversubscription
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        if None:
            use video resolution
        if (width, height) e.g. (1280, 720)
        if dict:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_decoding_threads <= 0:
        n_decoding_threads = multiprocessing.cpu_count()
    if n_encoding_threads <= 0:
        n_encoding_threads = multiprocessing.cpu_count()
    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)

    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    in_video_dir = input.joinpath('videos')
    assert in_zarr_path.is_dir()
    assert in_video_dir.is_dir()

    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')

    # save lowdim data to single chunk
    chunks_map = dict()
    compressor_map = dict()
    for key, value in in_replay_buffer.data.items():
        chunks_map[key] = value.shape
        compressor_map[key] = lowdim_compressor

    print('Loading lowdim data')
    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=in_replay_buffer.root.store,
        store=out_store,
        keys=lowdim_keys,
        chunks=chunks_map,
        compressors=compressor_map
    )

    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    n_cameras = 0
    camera_idxs = set()
    if image_keys is not None:
        n_cameras = len(image_keys)
        camera_idxs = set(int(x.split('_')[-1]) for x in image_keys)
    else:
        # estimate number of cameras
        episode_video_dir = in_video_dir.joinpath(str(0))
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        camera_idxs = set(int(x.stem) for x in episode_video_paths)
        n_cameras = len(episode_video_paths)

    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    print("episode_lengths", episode_lengths, n_steps)
    if dt is None:
        timestamps = in_replay_buffer['timestamp'][:]
        dt = timestamps[1] - timestamps[0]

    with tqdm(total=n_steps * n_cameras, desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
            futures = set()
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_video_dir = in_video_dir.joinpath(str(episode_idx))
                episode_start = episode_starts[episode_idx]

                episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
                this_camera_idxs = set(int(x.stem) for x in episode_video_paths)
                if image_keys is None:
                    for i in this_camera_idxs - camera_idxs:
                        print(f"Unexpected camera {i} at episode {episode_idx}")
                for i in camera_idxs - this_camera_idxs:
                    print(f"Missing camera {i} at episode {episode_idx}")
                    if image_keys is not None:
                        raise RuntimeError(f"Missing camera {i} at episode {episode_idx}")

                for video_path in episode_video_paths:
                    camera_idx = int(video_path.stem)
                    if image_keys is not None:
                        # if image_keys provided, skip not used cameras
                        if camera_idx not in camera_idxs:
                            continue

                    # read resolution
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        this_res = (vcc.width, vcc.height)
                    in_img_res = this_res

                    arr_name = f'camera_{camera_idx}'
                    # figure out save resolution
                    out_img_res = in_img_res
                    if isinstance(out_resolutions, dict):
                        if arr_name in out_resolutions:
                            out_img_res = tuple(out_resolutions[arr_name])
                    elif out_resolutions is not None:
                        out_img_res = tuple(out_resolutions)

                    # allocate array
                    if arr_name not in out_replay_buffer:
                        ow, oh = out_img_res
                        _ = out_replay_buffer.data.require_dataset(
                            name=arr_name,
                            shape=(n_steps, oh, ow, 3),
                            chunks=(1, oh, ow, 3),
                            compressor=image_compressor,
                            dtype=np.uint8
                        )
                    arr = out_replay_buffer[arr_name]

                    image_tf = get_image_transform(
                        input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)
                    for step_idx, frame in enumerate(read_video(
                        video_path=str(video_path),
                        dt=dt,
                        img_transform=image_tf,
                        thread_type='FRAME',
                        thread_count=n_decoding_threads
                    )):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures,
                                                                         return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        global_idx = episode_start + step_idx
                        futures.add(executor.submit(put_img, arr, global_idx, frame))

                        if step_idx == (episode_length - 1):
                            break
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))
    return out_replay_buffer


def create_zarr_action_dataset(dataset_path: str):
    dataset_path = pathlib.Path(dataset_path)
    action_path = dataset_path / "actions"

    zarr_path = str(dataset_path.joinpath('replay_buffer.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=zarr_path, mode='w')

    subfolders_actions = [file for file in action_path.iterdir()]

    for _subfolder in subfolders_actions:
        print(_subfolder)
        _file_path = _subfolder / "target_end_effector_pos_interpolated.npy"
        _file_path_eef = _subfolder / "current_eef_pos_interpolated.npy"
        _file_path_annotations = _subfolder / "annotations_video_interpolated.npy"
        _file_path_mass = _subfolder / "mass.txt"

        _file_path = _file_path.absolute()
        _file_path_eef = _file_path_eef.absolute()
        _file_path_annotations = _file_path_annotations.absolute()
        _file_path_mass = _file_path_mass.absolute()

        if not _file_path.exists():
            print(f"{_file_path} does not exist, skipping...")
            continue
        if not _file_path_eef.exists():
            print(f"{_file_path_eef} does not exist, skipping...")
            continue
        if not _file_path_annotations.exists():
            print(f"{_file_path_annotations} does not exist, skipping...")
            continue
        if not _file_path_mass.exists():
            print(f"{_file_path_mass} does not exist, skipping...")
            continue
        array_actions = np.load(_file_path)
        array_eef = np.load(_file_path_eef)
        array_annotations = np.load(_file_path_annotations)
        mass_scooped = np.loadtxt(_file_path_mass).item()
        array_masses = np.full((array_eef.shape[0],), fill_value=mass_scooped)

        if len(array_annotations) > len(array_masses):
            array_annotations = array_annotations[:len(array_masses)]

        print(array_actions.shape, array_eef.shape, array_annotations.shape, array_masses.shape)

        data_dict = {
            'action': array_actions,
            'eef': array_eef,
            'label': array_annotations,
            'mass': array_masses,
        }

        # print(data_dict, _subfolder)

        replay_buffer.add_episode(
            data_dict, compressors="disk",
        )


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, )
    return parser.parse_args()


def main():
    args = get_args()
    dataset_path = pathlib.Path(args.dataset_path)
    output_path = dataset_path / "replay_buffer_final.zarr.zip"
    assert output_path.suffix == ".zip"
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        create_zarr_action_dataset(dataset_path=dataset_path)
        replay_buffer = real_data_to_replay_buffer(dataset_path=dataset_path,
                                                   image_keys=tuple(f"{index}" for index in range(2)),
                                                   # 2 because there are 3 cameras
                                                   dt=0.1,
                                                   )

    with zarr.ZipStore(output_path) as zip_store:
        replay_buffer.save_to_store(
            store=zip_store
        )


if __name__ == '__main__':
    main()
