#!/usr/bin/python3
import socket
import logging
import struct
import numpy as np  
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ForceSensorController:
    # 类常量定义所有AT指令
    AT_GOD = "AT+GOD\r\n"  # 获取数据指令
    AT_EIP_QUERY = "AT+EIP=?\r\n"  # 查询IP地址
    AT_DCPM_QUERY = "AT+DCPM=?\r\n"  # 查询解耦矩阵
    AT_DCPCU_QUERY = "AT+DCPCU=?\r\n"  # 查询计算单位
    AT_SMPR_SET = "AT+SMPR={}\r\n"  # 设置采样频率
    AT_DCPCU_SET = "AT+DCPCU={}\r\n"  # 设置计算单位
    AT_SGDM_SET = "AT+SGDM=(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)\r\n"  # 设置上传数据格式
    AT_GSD_START = "AT+GSD\r\n"  # 启动数据流
    AT_GSD_STOP = "AT+GSD=STOP\r\n"  # 停止数据流
    
    def __init__(self, address=('192.168.0.108',4008),buffer_size=128):
        self._address = address
        self._buffer_size = buffer_size
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            self._socket.connect(self._address)
            logging.info(f"成功连接到传感器 {self._address[0]}:{self._address[1]}")
        except socket.error as e:
            logging.error(f"无法连接到传感器 {self._address[0]}:{self._address[1]}，错误：{e}")
        
        self._transform_matrix = self.set_decoupling_matrix()
        self._offset = self.sensor_data_init()
        # self._decoupling_matrix = self.get_decoupling_matrix()
        self._decoupling_matrix = self._transform_matrix
        
    def disconnect(self):
        self._socket.close()
        logging.info(f"成功断开与传感器 {self._address[0]}:{self._address[1]} 的连接")
    
    def send_command(self, command):
        try:
            self._socket.sendall(command.encode('utf-8'))
            logging.debug(f"发送指令：{command.strip()}")
            recv_data = self._socket.recv(self._buffer_size)
            logging.debug(f"接收数据：{recv_data}")
            return recv_data
        except socket.error as e:
            logging.error(f"发送指令失败：{e}")
            return None
        
    def sensor_data_init(self,init_times=666):
        self._sensor_data = [0.0] * 6
        # 偏移量校准
        raw_data_list = []
        for _ in range(666):
            raw_data = self.get_data()
            if raw_data is not None:
                raw_data_list.append(np.array(raw_data))
            
        if raw_data_list:
            self._offset = np.mean(raw_data_list, axis=0).tolist()
            logging.info(f"校准完成，偏移量为：{self._offset}")
            return self._offset
        else:
            logging.warning("未能获取到偏移量数据，使用默认偏移量。")
            return [0.0] * 6
                
    def get_data(self):
        data_raw = self.send_command(self.AT_GOD)
        if not data_raw or len(data_raw) < 30:
            logging.warning("接收到的数据长度不足，使用上一次数据")
            return self._sensor_data
        self._sensor_data = struct.unpack('6f',data_raw[6:30])
        return self._sensor_data
    
    def stop_data_stream(self):
        self.send_command(self.AT_GSD_STOP)
        logging.info("停止数据流")
        
    def set_decoupling_matrix(self):
        """构建坐标转换矩阵"""
        # 构建6x6转换矩阵
        transform_matrix = np.zeros((6, 6))
        
        # 设置矩阵元素
        cos_pi_6 = np.cos(np.pi / 6)  # 30度
        cos_pi_3 = np.cos(np.pi / 3)  # 60度
        
        # Fx行
        transform_matrix[0, 0] = -cos_pi_6
        transform_matrix[0, 1] = cos_pi_3
        
        # Fy行
        transform_matrix[1, 0] = -cos_pi_3
        transform_matrix[1, 1] = -cos_pi_6
        
        # Fz行
        transform_matrix[2, 2] = -1
        
        # Mx行
        transform_matrix[3, 3] = -cos_pi_6
        transform_matrix[3, 4] = cos_pi_3
        
        # My行
        transform_matrix[4, 3] = -cos_pi_3
        transform_matrix[4, 4] = -cos_pi_6
        
        # Mz行
        transform_matrix[5, 5] = -1
        
         # 将矩阵转换为字符串格式
        matrix_str = ""
        for i in range(6):
            row = transform_matrix[i]
            row_str = f"({','.join(f'{x:.5f}' for x in row)});"
            matrix_str += row_str
        matrix_str += "\r\n"
        
        # 构建设置指令
        command = f"AT+DCPM={matrix_str}"
        
        # 发送指令
        response = self.send_command(command)
        
        return transform_matrix
    
    # def get_decoupling_matrix(self):
    #     """获取解耦矩阵"""
    #     data_raw = self.send_command(self.AT_DCPM_QUERY)
    #     if not data_raw:
    #         logging.error("获取解耦矩阵失败")
    #         return None
        
    #     try:
    #         # 解析返回的数据，获取36个浮点数(6x6矩阵)
    #         matrix_data = struct.unpack('36f', data_raw[6:150])
    #         # 转换为numpy矩阵 6x6
    #         self._decoupling_matrix = np.array(matrix_data).reshape(6, 6)
    #         logging.info("成功获取解耦矩阵")
    #         return self._decoupling_matrix
    #     except Exception as e:
    #         logging.error(f"解析解耦矩阵失败: {e}")
    #         return None
        
    def transform(self,data = None):
        """使用解耦矩阵转换数据"""
        # 获取当前传感器数据
        if data is None:
            current_data = np.array(self._sensor_data) - np.array(self._offset)
        else:
            current_data = np.array(data) - np.array(self._offset)
        if self._decoupling_matrix is not None:
            self._transformed_data = np.dot(self._decoupling_matrix, current_data)
            return self._transformed_data
        
        self._transformed_data = np.dot(self._transform_matrix, current_data)
        return self._transformed_data.tolist()
    
    def filter(self, data, window_size=5):
        """
        均值滤波器
        :param data: 输入数据(可以是单个数据点或数据列表)
        :param window_size: 滤波窗口大小
        :return: 滤波后的数据
        """
        # 初始化滑动窗口
        if not hasattr(self, '_filter_window'):
            self._filter_window = []
        
        # 转换输入数据为numpy数组
        current_data = np.array(data)
        
        # 添加新数据到窗口
        self._filter_window.append(current_data)
        
        # 保持窗口大小
        if len(self._filter_window) > window_size:
            self._filter_window.pop(0)
        
        # 计算均值
        filtered_data = np.mean(self._filter_window, axis=0)
        
        return filtered_data.tolist()

    def get_filtered_data(self, type = 'mean',window_size=5):
        """
        获取滤波后的传感器数据
        :param window_size: 滤波窗口大小
        :return: 滤波后的数据
        """
        raw_data = self.get_data()
        if type == 'mean':
            return self.filter(raw_data, window_size)
        
    def plot_force_data(self, window_size=100, refresh_rate=0.1):
        """
        动态绘制力传感器数据
        :param window_size: 显示窗口大小(帧数)
        :param refresh_rate: 刷新率(秒)
        """
        import matplotlib.pyplot as plt
        from collections import deque
        
        # 初始化数据队列
        force_data = {
            'Fx': deque(maxlen=window_size),
            'Fy': deque(maxlen=window_size),
            'Fz': deque(maxlen=window_size),
            'Mx': deque(maxlen=window_size),
            'My': deque(maxlen=window_size),
            'Mz': deque(maxlen=window_size)
        }
        
        # 设置图形
        plt.ion()  # 开启交互模式
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 设置标题
        ax1.set_title('Force')
        ax2.set_title('Torque')
        
        # 初始化线条
        lines_force = ax1.plot([], [], 'r-', [], [], 'g-', [], [], 'b-')
        lines_torque = ax2.plot([], [], 'r-', [], [], 'g-', [], [], 'b-')
        
        # 设置图例
        ax1.legend(['Fx', 'Fy', 'Fz'])
        ax2.legend(['Mx', 'My', 'Mz'])
        
        try:
            while True:
                data = self.get_filtered_data()
                if data is not None:
                    transformed_data = self.transform(data)
                    
                    # 更新数据队列
                    for i, key in enumerate(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']):
                        force_data[key].append(transformed_data[i])
                    
                    # 更新力数据图
                    for i, line in enumerate(lines_force):
                        line.set_data(range(len(force_data['Fx'])), 
                                    list(force_data[['Fx', 'Fy', 'Fz'][i]]))
                    
                    # 更新力矩数据图
                    for i, line in enumerate(lines_torque):
                        line.set_data(range(len(force_data['Mx'])), 
                                    list(force_data[['Mx', 'My', 'Mz'][i]]))
                    
                    # 自动调整坐标轴
                    ax1.relim()
                    ax1.autoscale_view()
                    ax2.relim()
                    ax2.autoscale_view()
                    
                    # 更新显示
                    plt.draw()
                    plt.pause(refresh_rate)
                    
        except KeyboardInterrupt:
            logging.info("用户中断绘图")
        finally:
            plt.ioff()
            plt.close()
            
if __name__ == "__main__":
    sensor = ForceSensorController()
    try:
        sensor.plot_force_data()
    finally:
        sensor.stop_data_stream()
        sensor.disconnect()