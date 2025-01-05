import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseController:
    def __init__(self):
        # 初始化相机配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 获取设备信息
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        # 配置深度模块
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'Stereo Module':
                s.set_option(rs.option.exposure, 50000)
                found_rgb = True
                break
        if not found_rgb:
            raise RuntimeError("The demo requires Depth camera with Color sensor")

        # 配置数据流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动相机
        self.pipeline.start(self.config)

    def get_image(self, type='rgb'):
        """
        获取图像
        :param type: 'rgb' 或 'depth'
        :return: numpy格式的图像数据
        """
        # 等待获取图像帧
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None

        if type == 'rgb':
            return np.asanyarray(color_frame.get_data())
        elif type == 'depth':
            depth_image = np.asanyarray(depth_frame.get_data())
            # 将深度图转换为彩色图显示
            return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        else:
            raise ValueError("type must be 'rgb' or 'depth'")

    def get_aligned_frames(self):
        """
        获取对齐的RGB和深度图
        :return: RGB图像和深度图
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(
                np.asanyarray(depth_frame.get_data()), 
                alpha=0.03
            ), 
            cv2.COLORMAP_JET
        )

        return color_image, depth_image

    def close(self):
        """
        关闭相机
        """
        self.pipeline.stop()

if __name__ == "__main__":
    # 使用示例
    camera = RealSenseController()
    try:
        while True:
            # 获取RGB和深度图
            # color_image = camera.get_image('rgb')
            # depth_image = camera.get_image('depth')
            color_image, depth_image = camera.get_aligned_frames()
            if color_image is not None and depth_image is not None:
                # 水平拼接显示
                images = np.hstack((color_image, depth_image))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.close()
        cv2.destroyAllWindows()