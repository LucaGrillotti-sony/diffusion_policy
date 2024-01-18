import numpy as np
import quaternion as quat

def quat_library_from_ros(q):
    q = q.ravel()
    return np.asarray([q[3], q[0], q[1], q[2]]).ravel()

def quat_ros_from_library(q):
    q = q.ravel()
    return np.asarray([q[1], q[2], q[3], q[0]]).ravel()
