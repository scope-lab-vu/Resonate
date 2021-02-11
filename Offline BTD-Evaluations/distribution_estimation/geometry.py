import numpy as np


def calc_angle(v1, v2):
    return np.arccos(np.vdot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


if __name__ == "__main__":
    """Simple example/test"""
    vec1 = [0, 1, 0]
    vec2 = [1, 1, 0]
    vec3 = [1, 0, 0]
    print("angle (deg): ", np.rad2deg(calc_angle(vec1, vec3)))
