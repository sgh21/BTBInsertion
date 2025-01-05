import ur_control as UR
import mvs_control as MVS
import gelsight_control as GelSight
import realsense_control as RealSense
import force_control as ForceControl
import cv2
import numpy as np
    
def test_mvs():
    mvs_control = MVS.MVSController()
    mvs_control.init()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1024, 1224)
    while True:
        image = mvs_control.get_image()
        if image is not None:
            print(image.shape)
            height, width = image.shape[:2]
            center_point = (width // 2, height // 2)
            cv2.circle(image, center_point, 10, (0, 255, 0), 2)
            cv2.imshow("image", image)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    mvs_control.close_device()
    cv2.destroyAllWindows()
    print("done")

def test_gelsight():
    gelsight = GelSight.GelSightController()
    
    print('press q on image to exit, press s to save image')
    try:
        while gelsight.dev.while_condition:
            f1 = gelsight.get_image()
            # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            f2 = cv2.resize(f1, (640, 480))
            cv2.imshow('Image', f2)
            # if cv2.waitKey(1) & 0xFF == ord('s'):
            #     cv2.imwrite(f'../result/gelsightmini/tactile_image_{i}.png', f1)
            #     print(f'Image saved_{i}')
            #     i = i + 1

            #     time.sleep(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('Interrupted!')
        gelsight.dev.stop_video()

    cv2.destroyAllWindows()
def test_realsense():
    realsense = RealSense.RealSenseController()
    try:
        while True:
            # 获取RGB和深度图
            # color_image = camera.get_image('rgb')
            # depth_image = camera.get_image('depth')
            color_image, depth_image = realsense.get_aligned_frames()
            if color_image is not None and depth_image is not None:
                # 水平拼接显示
                images = np.hstack((color_image, depth_image))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        realsense.close()
        cv2.destroyAllWindows()
    
def test_ur():
    hostname = '192.168.0.10'
    timeout = 2
    ur = UR.URController(hostname)
    while not ur.connect() and timeout > 0:
        print('Connecting...')
        timeout -= 1
    else:
        print('Connected')
    
    print("getJointTorques",ur.getJointTorques())
    print("getActualTCPPose",ur.getActualTCPPose())
    print("getActualTCPSpeed",ur.getActualTCPSpeed())
    print("getActualTCPForce",ur.getActualTCPForce())
    print("getTargetTCPPose",ur.getTargetTCPPose())
    print("getTargetTCPSpeed",ur.getTargetTCPSpeed())
    print("getActualToolAccelerometer",ur.getActualToolAccelerometer())
    
    current_q = ur.getActualQ()
    current_p = ur.getActualTCPPose()
    print("current_q",current_q)
    print("current_p",current_p)
    delta_p = [0.001, 0, 0, 0, 0, 0]
    target_p = [current_p[i]+delta_p[i] for i in range(6)]
    target_q = ur.getInverseKinematics(target_p,current_q)
    print("target_q",target_q)
    # target_q = [current_q[i]+delta_q[i] for i in range(6)]
    # ur.moveJ(target_q)
    ur.disconnect()
    
def test_forece_control():
    force_control = ForceControl.ForceSensorController()
    try:
        force_control.plot_force_data()
    finally:
        force_control.stop_data_stream()
        force_control.disconnect()
if __name__ == "__main__":
    # test_gelsight()
    test_mvs()
    # test_realsense()
    # test_ur()
    # import sys
    # paths = sys.path
    # for path in paths:
    #     print(path)
    # test_forece_control()
    