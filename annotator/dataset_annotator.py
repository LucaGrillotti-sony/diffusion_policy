import cv2
import matplotlib.pyplot as plt
import numpy as np

from annotator.utils import interpolate


def annotate_video(video_path, save_path, number_total_actions=None, new_fps=10,):
    import cv2
    import sys

    # load input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("!!! Failed cap.isOpened()")
        sys.exit(-1)

    # retrieve the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    window_name = video_path.name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # print(fps, new_fps)
    # new_timesteps = np.arange(0, frame_count / fps, step=1./new_fps)

    # assert len(new_timesteps) == number_total_actions, f"len(new_timesteps) {len(new_timesteps)} != number_total_actions {number_total_actions}"

    is_scooped_labels = []

    index = 0
    # loop to read every frame of the video
    while cap.isOpened():

        # capture a frame
        ret, frame = cap.read()
        if not ret:
            print("!!! Failed cap.read()")
            break

        cv2.imshow(window_name, frame)

        # check if 'p' was pressed and wait for a 'b' press
        key = cv2.waitKey()
        if key & 0xFF == ord('p'):

            # sleep here until a valid key is pressed
            while True:
                key = cv2.waitKey(0)

                # check if 'p' is pressed and resume playing
                if key & 0xFF == ord('p'):
                    break

                # check if 'b' is pressed and rewind video to the previous frame, but do not play
                if key & 0xFF == ord('b'):
                    cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    print('* At frame #' + str(cur_frame_number))

                    prev_frame = cur_frame_number
                    if cur_frame_number > 1:
                        prev_frame -= 1

                    print('* Rewind to frame #' + str(prev_frame))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)

                # check if 'r' is pressed and rewind video to frame 0, then resume playing
                if key & 0xFF == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break

        # exit when 'q' is pressed to quit
        elif key & 0xFF == ord('q'):
            break

        elif key & 0xFF == ord('0'):
            is_scooped_labels.append(0)
            print("0")
        elif key & 0xFF == ord('1'):
            is_scooped_labels.append(1)
            print("1")
        elif key & 0xFF == ord('2'):
            is_scooped_labels.append(2)
            print("2")

        index += 1

    array_of_labels = np.asarray(is_scooped_labels)
    assert len(array_of_labels) == frame_count
    # current_timesteps = np.arange(0, array_of_labels.shape[0] / fps, step=1./fps)
    # new_fps = 10
    # new_timesteps = np.arange(0, array_of_labels.shape[0] / fps, step=1./new_fps)

    # assert len(new_timesteps) == number_total_actions

    # booleans_interpolated = interpolate(x=current_timesteps, y=array_of_labels, new_x=new_timesteps)
    with open(str(save_path), "wb") as f:
        np.save(f, array_of_labels)
        print(f"saving video to {save_path}")

    # release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Replace 'your_video.mp4' with the path to your video file
    video_path = 'your_video.mp4'
    save_path = 'is_scooped_booleans.npy'
    annotate_video(video_path, save_path, number_total_actions=299, new_fps=10)


if __name__ == '__main__':
    main()
