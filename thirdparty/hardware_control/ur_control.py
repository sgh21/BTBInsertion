import rtde_receive
import rtde_control 

class URController:
   
    def __init__(self, hostname,**kwargs):
        
        self._hostname = hostname
        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):

        self._blend = kwargs.get('blend', 0.001)
        self._rtde_freq = kwargs.get('rtde_freq', 500)
        self._max_acc = kwargs.get('max_acc', 0.3)
        self._max_vel = kwargs.get('max_vel', 0.1)

    def connect(self):
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self._hostname)
        self._rtde_c = rtde_control.RTDEControlInterface(self._hostname, \
                                                         self._rtde_freq,\
                                                         rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,\
                                                         30004)
        
        return self._rtde_r.isConnected() and self._rtde_c.isConnected()
    
    def disconnect(self):
        self._rtde_c.disconnect()
        self._rtde_r.disconnect()
    
    def getJointTorques(self):
        return self._rtde_c.getJointTorques()

    def getActualQ(self):
        return self._rtde_r.getActualQ()
    
    def getTargetQd(self):
        return self._rtde_r.getTargetQd()
    
    def getActualCurrent(self):
        return self._rtde_r.getActualCurrent()
    
    def getJointControlOutput(self):
        return self._rtde_r.getJointControlOutput()
    
    def getActualToolFlangePose(self):
        return self._rtde_c.getActualToolFlangePose()
    
    def getActualTCPPose(self):
        return self._rtde_r.getActualTCPPose()
    
    def getActualTCPSpeed(self):
        return self._rtde_r.getActualTCPSpeed()
    
    def getActualTCPForce(self):
        return self._rtde_r.getActualTCPForce()
    
    def getTargetTCPPose(self):
        return self._rtde_r.getTargetTCPPose()
    
    def getTargetTCPSpeed(self):
        return self._rtde_r.getTargetTCPSpeed()
    
    def getActualToolAccelerometer(self):
        return self._rtde_r.getActualToolAccelerometer()
    
    def getTCPOffset(self):
        return self._rtde_c.getTCPOffset()
    
    def moveJ(self,q,speed = 1.05, acceleration = 1.4, asynchronous = False):
        self._rtde_c.moveJ(q, speed, acceleration, asynchronous)
    
    def moveJ_IK(self,pose,speed = 1.05, acceleration = 1.4, asynchronous = False):
        self._rtde_c.moveJ_IK(pose, speed, acceleration, asynchronous)
        
    def moveL(self,pose,speed = 0.25, acceleration = 1.2, asynchronous = False):
        self._rtde_c.moveL(pose, speed, acceleration, asynchronous)
        
    def moveL_FK(self,q,speed = 0.25, acceleration = 1.2, asynchronous = False):
        self._rtde_c.moveL_FK(q, speed, acceleration, asynchronous)
    
    def speedJ(self,qd, acceleration = 0.5, time= 0.0):
        """
        非阻塞式关节速度控制

        Args:
            qd: joint speeds [rad/s]
            acceleration: joint acceleration [rad/s^2] (of leading axis)
            time: time [s] before the function returns (optional)
        """        
        
        self._rtde_c.speedJ(qd, acceleration,time)
    
    def speedL(self,vd, acceleration = 0.25, time= 0.0):
        """
        非阻塞式末端速度控制

        Args:
            xd: tool speed [m/s] (spatial vector)
            acceleration: tool position acceleration [m/s^2]
            time: time [s] before the function returns (optional)
        """        
        self._rtde_c.speedL(vd, acceleration,time)
        
    def servoJ(self,q,speed,acceleration,time,lookahead_time=0.1,gain=300):
        """
        非阻塞式关节伺服控制

        Args:
            q: joint positions [rad]
            speed: NOT used in current version
            acceleration: NOT used in current version
            time: time [s] before the function returns
            lookahead_time: lookahead time [s]
            gain: servo gain
        """
        self._rtde_c.servoJ(q,speed,acceleration,time,lookahead_time,gain)
        
    def servoL(self,pose,speed,acceleration,time,lookahead_time=0.1,gain=300):
        """
        Servo to position (linear in tool-space)

        Args:
            pose: target pose
            speed: NOT used in current version
            acceleration: NOT used in current version
            time: time where the command is controlling the robot. The function is blocking for time t [S]
            lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
            gain: proportional gain for following target position, range [100,2000]
        """
        return self._rtde_c.servoL(pose,speed,acceleration,time,lookahead_time,gain)
    
    def setPayload(self,mass,CoM):
        """
        设置负载

        Args:
            mass: 负载质量 [kg]
            CoM: 负载质心坐标 [m]
        """
        self._rtde_c.setPayload(mass,CoM)
        
    def setTcp(self, tcp_offset):
        """
        设置工具中心点想对于法兰的偏移

        Args:
            tcp_offset: A pose describing the transformation of the tcp offset.
        """
        self._rtde_c.setTcp(tcp_offset)
        
    def getInverseKinematics(self,pose,qnear = None, maxPositionError = 1e-6, maxOrientationError = 1e-6):
        """
        Calculate the inverse kinematic transformation (tool space -> jointspace).

        If qnear is defined, the solution closest to qnear is returned.Otherwise, 
        the solution closest to the current joint positions is returned. 
        If no tcp is provided the currently active tcp of the controller will be used.

        Args:
            pose: tool pose
            qnear: list of joint positions (Optional)
            maxPositionError: the maximum allowed positionerror (Optional)
            maxOrientationError: the maximum allowed orientationerror (Optional)

        Returns:
            list: joint positions
        """
        return self._rtde_c.getInverseKinematics(pose,qnear,maxPositionError,maxOrientationError)
    
    def poseTrans(self,pose_from,pose_to):
        """
        Calculate the transformation between two poses.

        Args:
            pose_from: the start pose
            pose_to: the target pose

        Returns:
            list: the transformation pose
        """
        return self._rtde_r.poseTrans(pose_from,pose_to)
    
    
if __name__ == '__main__':
    hostname = '192.168.0.10'
    timeout = 2
    ur = URController(hostname)
    while not ur.connect() and timeout > 0:
        print('Connecting...')
        timeout -= 1
    else:
        print('Connected')
    import numpy as np
    from utils.transform import *
    robot_init_pose = [-107,-549,262.5,-180,0,90]
    robot_init_pose[:3] = np.array(robot_init_pose[:3])/1000
    init_pose_rpy = robot_init_pose[3:] = np.array(robot_init_pose[3:])*np.pi/180
    
    init_pose_axisangle = rpy2axisangle(init_pose_rpy)
    robot_init_pose[3:] = init_pose_axisangle
    print("The init pose of the robot is:",robot_init_pose)
    
    ur.moveL(robot_init_pose)
    ur.disconnect()