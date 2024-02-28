import copy
import pathlib
import sqlite3

import numpy as np
import matplotlib.pyplot as plt

from std_msgs.msg import String


from rosidl_runtime_py.utilities import get_message

get_message("std_msgs/msg/String")
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import PyKDL
from kdl_solver import KDLSolver


class BagFileParser():

    CONVERSION_MODULES = {
        "String": "std_msgs/msg/String",
        "FrankaRobotState": "franka_msgs/msg/FrankaRobotState",
        "CompressedImage": "sensor_msgs/msg/CompressedImage",
    }

    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}

        for key, value in self.topic_type.items():
            if value in self.CONVERSION_MODULES:
                self.topic_type[key] = self.CONVERSION_MODULES[value]

        for topic_value in self.topic_type.values():
            print(topic_value)
            get_message(topic_value)

        self.topic_msg_message = {name_of: get_message(self.topic_type[name_of]) for id_of, name_of, type_of in topics_data}
        print("topic id", self.topic_id)

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):
        print("TOPICs", list(self.topic_id.keys()))
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]


def collect_and_save_data(path_load, path_save):
    path_load = pathlib.Path(path_load)
    path_save = pathlib.Path(path_save)

    path_save.mkdir(parents=True, exist_ok=True)

    folder_name = path_load.name
    bag_file = f'{path_load}/{folder_name}_0.db3'

    parser = BagFileParser(bag_file)

    # trajectory = parser.get_messages("/franka_robot_state_broadcaster/robot_state")[0][1]
    trajectory = parser.get_messages("/franka_robot_state_broadcaster/robot_state")
    trajectory_joint_states = parser.get_messages("/joint_states")
    print("len traj", len(trajectory))
    msg_dq = parser.get_messages("/dq")
    # msg_cartesian = parser.get_messages("/cartesian_control")

    # Save DQ with time
    all_dq = [
        np.asarray(dq[1].data)
        for dq in msg_dq
    ]
    all_dq = np.asarray(all_dq)

    # all_cartesian_control = [
    #     np.asarray(command[1].data)
    #     for command in msg_cartesian
    # ]

    all_times_dq = np.asarray(
        [dq[0] / 1e9 for dq in msg_dq]
    )
    all_times_dq = all_times_dq - all_times_dq[0]

    # all_times_cartesian_control = np.asarray(
    #     [_msg[0] / 1e9 for _msg in msg_cartesian]
    # )
    # all_times_cartesian_control = all_times_cartesian_control - all_times_cartesian_control[0]

    # plt.plot(all_times_dq, all_dq)
    # plt.show()

    dq_with_time = np.hstack([all_times_dq.reshape(-1, 1), all_dq])
    np.save(path_save / "dq_with_time.npy", dq_with_time)

    # cartesian_control_with_time = np.hstack([all_times_cartesian_control.reshape(-1, 1), all_cartesian_control])
    # np.save(path_save / "cartesian_control_with_time.npy", cartesian_control_with_time)

    init_joint_pos = np.asarray(trajectory[0][1].q)
    np.save(path_save / "init_joint_pos.npy", init_joint_pos)

    # Get and save Obs with Time
    robot_description = parser.get_messages("/robot_description")[-1][1].data
    kdl = KDLSolver(robot_description)
    kdl.set_kinematic_chain('panda_link0', 'panda_hand')

    all_end_effector_poses = []
    all_joint_poses = []
    all_times_cartesian_commands = []
    for command in trajectory_joint_states:
        time_cartesian_commands = command[0] / 1e9
        pos_joints = command[1].position[:7]
        pos_end_effector_xyz, pos_end_effector_rot = kdl.compute_fk(np.asarray(pos_joints))
        pos_end_effector = np.concatenate([pos_end_effector_xyz.ravel(), pos_end_effector_rot.ravel()])
        all_end_effector_poses.append(pos_end_effector)
        all_times_cartesian_commands.append(time_cartesian_commands)
        all_joint_poses.append(np.asarray(pos_joints))
    all_end_effector_poses = np.asarray(all_end_effector_poses)
    all_times_cartesian_commands = np.asarray(all_times_cartesian_commands) - all_times_cartesian_commands[0]
    all_joint_poses = np.asarray(all_joint_poses)

    all_qpos = [
        np.asarray([*point[1].q, 0., 0.]) for point in trajectory
    ]
    all_qpos = np.asarray(all_qpos)

    all_times_obs = [
        point[1].time for point in trajectory
    ]
    all_times_obs = np.asarray(all_times_obs) - all_times_obs[0]
    obs_with_time = np.hstack([all_times_obs.reshape(-1, 1), all_qpos])
    end_effector_poses_with_time = np.hstack([all_times_cartesian_commands.reshape(-1, 1), all_end_effector_poses])
    end_effector_poses_xyz_with_time = end_effector_poses_with_time[:, :4]
    all_joint_poses_with_time = np.hstack([all_times_cartesian_commands.reshape(-1, 1), all_joint_poses])

    np.save(path_save / "obs_with_time.npy", obs_with_time)
    np.save(path_save / "end_effector_poses_with_time.npy", end_effector_poses_with_time)
    np.save(path_save / "end_effector_poses_xyz_with_time.npy", end_effector_poses_xyz_with_time)
    np.save(path_save / "all_joint_poses_with_time.npy", all_joint_poses_with_time)


def main():
    # folder_name = "rosbag2_2023_12_07-18_45_17"
    folder_to_parse = pathlib.Path("/home/ros/humble/saved/")
    path_save = folder_to_parse / "/home/ros/humble/src/read_db/extracted_obs_actions/cartesian_control/"
    subdirs = [x for x in folder_to_parse.iterdir() if x.is_dir() and x.name.startswith("rosbag2_")]
    for _subdir in subdirs:
        print(f"Loading {_subdir}")
        name_subdir = _subdir.name
        _path_save = path_save / name_subdir
        collect_and_save_data(
            path_load=_subdir,
            path_save=_path_save,
        )


if __name__ == '__main__':
    main()
