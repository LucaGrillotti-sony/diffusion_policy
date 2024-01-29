import scipy

def interpolate(x, y, new_x):
    interpolator = scipy.interpolate.interp1d(x, y, axis=0,
                                              kind="nearest", fill_value="extrapolate", )
    target_end_effector_pos_interpolated = interpolator(new_x)
    return target_end_effector_pos_interpolated