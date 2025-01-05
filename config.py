import os

WORK_DIR  = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(WORK_DIR, 'weights')
# Yolo for object detection
PLUG_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'connector_plug.pt')
EXPERIMENT_DIR = os.path.join(WORK_DIR, 'experiments')
DATA_DIR = os.path.join(WORK_DIR, 'documents')
# 在相机上的初始位姿势[x,y,z,r,p,y] mm,deg
# ! 请注意：这里的单位与机器人控制时并不统一，只是为了便于调试
ROBOT_INIT_POSE = [-107,-549,262.5,-180,0,90]
# 相机内参 rx = focal_length / pixel_width
# 相机内参 ry = focal_length / pixel_height
INTRINSIC = [-0.0206,-0.0207]
# HSV颜色空间的下限和上限
HSV_LOWER_BOUND = [0, 0, 0]
HSV_UPPER_BOUND = [180, 255, 55]

PARAMS = {
  'work_dir': WORK_DIR,
  'weights_dir': WEIGHTS_DIR,
  'yolo_model_path': PLUG_MODEL_PATH,
  'robot_init_pose': ROBOT_INIT_POSE,
  'experiment_dir': EXPERIMENT_DIR,
  'data_dir': DATA_DIR,
  'intrinsic': INTRINSIC,
  'hsv_lower_bound': HSV_LOWER_BOUND,
  'hsv_upper_bound': HSV_UPPER_BOUND
}

print("All parameters are loaded successfully!")
# print(PARAMS)