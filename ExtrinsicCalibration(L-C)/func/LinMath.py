import numpy as np
from scipy.spatial.transform import Rotation as R

def rotationMatrixToEulerAngles(matrix: np.array, type: str):
    # matrix: 3x3 rotation matrix
    # type: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    # return: euler angles
    r = R.from_matrix(matrix)
    return r.as_euler(type, degrees=False)

def eulerAnglesToRotationMatrix(rot: np.array, type: str):
    # rot: radian euler angles
    # type: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    # return: 3x3 rotation matrix
    r = R.from_euler(type, rot, degrees=False)
    return r.as_matrix()
