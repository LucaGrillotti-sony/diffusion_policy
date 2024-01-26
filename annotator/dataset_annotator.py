import cv2
import matplotlib.pyplot as plt
import numpy as np

from read_sensors_utils.format_data_replay_buffer import interpolate


def show_video(video_path, save_path, number_total_actions=None, new_fps=10):
    import cv2
    import sys

    # load input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("!!! Failed cap.isOpened()")
        sys.exit(-1)

    # retrieve the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    new_timesteps = np.arange(0, frame_count / fps, step=1./new_fps)

    assert len(new_timesteps) == number_total_actions

    is_scooped_booleans = []

    index = 0
    # loop to read every frame of the video
    while cap.isOpened():

        # capture a frame
        ret, frame = cap.read()
        if not ret:
            print("!!! Failed cap.read()")
            break

        cv2.imshow('video', frame)

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
            is_scooped_booleans.append(False)
            print("0")
        elif key & 0xFF == ord('1'):
            is_scooped_booleans.append(True)
            print("1")

        index += 1

    array_of_booleans = np.asarray(is_scooped_booleans)
    current_timesteps = np.arange(0, array_of_booleans.shape[0] / fps, step=1./fps)
    new_fps = 10
    new_timesteps = np.arange(0, array_of_booleans.shape[0] / fps, step=1./new_fps)

    assert len(new_timesteps) == number_total_actions

    booleans_interpolated = interpolate(x=current_timesteps, y=array_of_booleans, new_x=new_timesteps)

    np.save(save_path, booleans_interpolated)

    # release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Replace 'your_video.mp4' with the path to your video file
    video_path = 'your_video.mp4'
    save_path = 'is_scooped_booleans.npy'
    show_video(video_path, save_path)


if __name__ == '__main__':
    main()
