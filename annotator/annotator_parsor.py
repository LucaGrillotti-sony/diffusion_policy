from pathlib import Path

import numpy as np

from annotator.dataset_annotator import annotate_video
from read_sensors_utils.format_data_replay_buffer import CURRENT_EEF_POS_INTERPOLATED

# Name file hand-eye video
HAND_EYE_VIDEO_NAME = "1.mp4"
NEW_FPS = 10

def run(folder_to_parse):
    folder_to_parse = Path(folder_to_parse)
    assert folder_to_parse.exists()

    path_actions = folder_to_parse / "actions"
    path_videos = folder_to_parse / "videos"

    for _subpath_action, _subpath_video in zip(sorted(path_actions.iterdir()), sorted(path_videos.iterdir())):
        _path_video_handeye = _subpath_video / HAND_EYE_VIDEO_NAME
        assert _path_video_handeye.exists()

        _path_eef_data = _subpath_action / CURRENT_EEF_POS_INTERPOLATED
        assert _path_eef_data.exists()

        eef_array = np.load(_path_eef_data)
        number_total_actions = eef_array.shape[0]

        print("Annotating video...", _path_video_handeye)
        input("Press enter to continue...")

        _path_save = _subpath_action / "annotations_video.npy"

        annotate_video(_path_video_handeye, _path_eef_data, number_total_actions=number_total_actions, new_fps=NEW_FPS)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder-to-parse", type=str, required=True)
    args = parser.parse_args()
    folder_to_parse = args.folder_to_parse

    run(folder_to_parse)


if __name__ == "__main__":
    main()


