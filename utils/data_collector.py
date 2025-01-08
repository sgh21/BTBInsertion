# 采集Yolo分割后的图像，用于进行引脚网络的训练
# TODO： 1.控制机械臂进行一定范围内的平移和旋转
# TODO： 2.使用YOLO进行目标检测，并记录剪切+缩放后的图像

from config import PARAMS
from utils.transform import *
from utils.yolo_dection import YOLODetection
import ur_control  as UR
import mvs_control as MVS
from tranditional_vision import *
import numpy as np
import pandas as pd
import os
import time

def random_pose(robot_init_pose,bound = (0.01,0.01,5/180*np.pi)):
    x = np.random.uniform(-bound[0],bound[0])
    y = np.random.uniform(-bound[1],bound[1])
    z = 0
    Rx = 0
    Ry = 0
    Rz = np.random.uniform(-bound[2],bound[2])
   
    delta_pose = np.array([x,y,z,Rx,Ry,Rz])
    
    new_pose = robot_init_pose.copy() 
    new_pose[:3] += delta_pose[:3]
    new_pose[3:] = rpy2axisangle(axisangle2rpy(robot_init_pose[3:])+delta_pose[3:])
    
    return new_pose


    
def main():
    hostname = '192.168.0.10'
    serial_type = '4040P'
    ur = UR.URController(hostname)
    mvs_control = MVS.MVSController()
    mvs_control.init()
    yolo = YOLODetection(PARAMS['yolo_model_path'])
    data_dir = PARAMS['data_dir']
    
    # 创建CSV文件，并记录数据
    csv_file = os.path.join(data_dir, f'tcp_position_{serial_type}.csv')
    columns = ['tcp_x', 'tcp_y', 'tcp_rz']
    csv_data = []
    
    if ur.connect():
        print("Connected")
    else:
        raise Exception("Connection failed")
    tcp_offset = [0,0,0.217,0,0,0]
    ur.setTcp(tcp_offset)
    robot_init_pose = PARAMS['robot_init_pose']
    # ! 请注意：这里的单位与机器人控制时并不统一，必须转换
    robot_init_pose[:3] = np.array(robot_init_pose[:3])/1000
    init_pose_rpy = robot_init_pose[3:] = np.array(robot_init_pose[3:])*np.pi/180
    
    init_pose_axisangle = rpy2axisangle(init_pose_rpy)
    robot_init_pose[3:] = init_pose_axisangle
    print("The init pose of the robot is:",robot_init_pose)
    
    ur.moveL(robot_init_pose)
    
    image = os.path.join(data_dir, f'image_{serial_type}')
    image_result = os.path.join(data_dir, f'image_result_{serial_type}')
    if not os.path.exists(image):
        os.makedirs(image)
    if not os.path.exists(image_result):
        os.makedirs(image_result)
    counter = 200
    for i in range(300):
    # while True:
        time.sleep(1)
        counter += 1
        img = mvs_control.get_image()
        img_path = os.path.join(image, f'image_{serial_type}_{counter}.png')
        cv2.imwrite(img_path, img)
        yolo_result = yolo.predict(img, padding=30, area_threshold=1e2)
        cut_img_list = yolo.cut_image()
        
        cut_img = cut_img_list[0]
        img_roi = cut_img.img.copy()
        img_scaled = scale_img(img_roi, scale_factor=2.0,interpolation=cv2.INTER_CUBIC)

        current_p = ur.getActualTCPPose()
        current_p[3:] = axisangle2rpy(current_p[3:])
        print("The current pose of the robot is:",current_p)
        current_p[3:] = np.array(current_p[3:])*180/np.pi
        
        
        raw_data = {
            'tcp_x': current_p[0],
            'tcp_y': current_p[1],
            'tcp_rz': current_p[5],
        }
    
        csv_data.append(raw_data)
        df = pd.DataFrame(csv_data, columns=columns)
        df.to_csv(csv_file, index=False)
        result_path = os.path.join(image_result, f'roi_image_{serial_type}_{counter}.png')
        
        # 保存图像
        cv2.imwrite(result_path, img_scaled)
        # cv2.imshow('roi', img_scaled)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        
        target_p = random_pose(robot_init_pose,bound=(0.013,0.013,10/180*np.pi))
        print("The target pose of the robot is:",target_p)
        
        ur.moveL(target_p)
        
    ur.moveL(robot_init_pose)
        

if __name__ == "__main__":
    main()