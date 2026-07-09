import numpy as np
import copy


def wrapToPi(rad):
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap


def wrapTo2Pi(rad):
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad


def smart_unwrap(angles, tolerance=0.01):
    """ Smart unwrapping function that deals with initial angles near pi or -pi.
    """
    init_angle = np.abs(angles[0])
    if (init_angle - np.pi) < tolerance:  # close to pi or - pi, wrap to 2pi
        angles_wrap = wrapToPi(angles)

    elif ((init_angle - 2*np.pi) < tolerance) or ((init_angle - 0.0) < tolerance):   # close to 2pi or 0, wrap to pi
        angles_wrap = wrapToPi(angles)

    else:  # leave as is
        angles_wrap = angles

    angles_unwrap = np.unwrap(angles_wrap)

    return angles_unwrap


def range_of_vals(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)
