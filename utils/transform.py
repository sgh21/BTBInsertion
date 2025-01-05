from scipy.spatial.transform import Rotation as R


def rpy2rot(rpy):
    """
    Convert roll-pitch-yaw angles to rotation matrix.
    
    Args:
        rpy: roll-pitch-yaw angles [rad]
    """
    r = R.from_euler('xyz', rpy)
    return r.as_matrix()

def rot2rpy(rot):
    """
    Convert rotation matrix to roll-pitch-yaw angles.
    
    Args:
        rot: rotation matrix
    """
    r = R.from_matrix(rot)
    return r.as_euler('xyz')

def rot2quat(rot):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        rot: rotation matrix
    """
    r = R.from_matrix(rot)
    return r.as_quat()

def quat2rot(quat):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quat: quaternion
    """
    r = R.from_quat(quat)
    return r.as_matrix()

def rpy2quat(rpy):
    """
    Convert roll-pitch-yaw angles to quaternion.
    
    Args:
        rpy: roll-pitch-yaw angles [rad]
    """
    r = R.from_euler('xyz', rpy)
    return r.as_quat()

def quat2rpy(quat):
    """
    Convert quaternion to roll-pitch-yaw angles.
    
    Args:
        quat: quaternion
    """
    r = R.from_quat(quat)
    return r.as_euler('xyz')

def rot2axisangle(rot):
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        rot: rotation matrix
    """
    r = R.from_matrix(rot)
    return r.as_rotvec()

def axisangle2rot(axisangle):
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axisangle: axis-angle representation
    """
    r = R.from_rotvec(axisangle)
    return r.as_matrix()

def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        quat: quaternion
    """
    r = R.from_quat(quat)
    return r.as_rotvec()

def axisangle2quat(axisangle):
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axisangle: axis-angle representation
    """
    r = R.from_rotvec(axisangle)
    return r.as_quat()

def rpy2axisangle(rpy):
    """
    Convert roll-pitch-yaw angles to axis-angle representation.
    
    Args:
        rpy: roll-pitch-yaw angles [rad]
    """
    r = R.from_euler('xyz', rpy)
    return r.as_rotvec()

def axisangle2rpy(axisangle):
    """
    Convert axis-angle representation to roll-pitch-yaw angles.
    
    Args:
        axisangle: axis-angle representation
    """
    r = R.from_rotvec(axisangle)
    return r.as_euler('xyz')

if __name__ =="__main__":
    euler = [-3.1415926,0,1.57]
    rot = rpy2rot(euler)
    print(rot)
    euler = rot2rpy(rot)
    print(euler)
    quat = rpy2quat(euler)
    print(quat)
    euler = quat2rpy(quat)
    print(euler)
    axisangle = rpy2axisangle(euler)
    print(axisangle)
    euler = axisangle2rpy(axisangle)
    print(euler)
    
    axisangle = [2.222,2.222,0]
    rot = axisangle2rot(axisangle)
    print(rot)
    axisangle = rot2axisangle(rot)
    print(axisangle)
    euler = axisangle2rpy(axisangle)
    print(euler)
    axisangle = rpy2axisangle(euler)
    print(axisangle)
    quat = axisangle2quat(axisangle)
    print(quat)
    axisangle = quat2axisangle(quat)
    print(axisangle)
