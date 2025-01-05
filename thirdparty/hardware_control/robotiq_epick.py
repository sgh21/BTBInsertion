# doc: https://assets.robotiq.com/website-assets/support_documents/document/EPick_Instruction_Manual_e-Series_PDF_20210709.pdf

import socket
import threading
import time

class EPickGripper:
    # 寄存器定义
    ACT = 'ACT'  # 激活
    GTO = 'GTO'  # 抓取(1)/释放(0)
    MOD = 'MOD'  # 模式(0: 自动, 1: 手动)
    MAX_VACUUM = 'POS'  # 最大真空度(0-100, 100为持续吸气，噪声较大)
    MIN_VACUUM = 'FOR'  # 最小真空度(0-100, 需小于最大真空度)
    TIMEOUT = 'SPE'  # 时间间隔(0-255, 单位: 0.1秒，该时间内未检测到对象则停止)
    OBJ = 'OBJ'  # 物体检测状态
    CURRENT_VACUUM = 'POS'  # 当前真空度
    STATUS = 'STA'  # 当前状态

    ENCODING = 'UTF-8'

    def __init__(self, ip):
        """EPick Gripper控制器的构造函数。"""
        self.socket = None
        self.command_lock = threading.Lock()
        self.connect(ip, 63352)
        self.set_mode(0)  # 设置为自动模式
        time.sleep(0.2)
        self.set_mode(1)  # 设置为手动模式
        # self.set_timeout(30)  # 设置超时时间为3秒
        self.set_vacuum_levels(75, 55)
        self.release()

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0):
        """连接ePick电动吸盘。"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((hostname, port))
            self.socket.settimeout(socket_timeout)
            print("已连接至ePick电动吸盘。")
        except (socket.error, socket.timeout) as e:
            print(f"错误：无法连接到电动吸盘: {e}")
            raise

    def disconnect(self):
        """断开与电动吸盘的连接。"""
        if self.socket:
            self.socket.close()
            print("已断开与电动吸盘的连接。")

    def set_mode(self, mode: int):
        """设置电动吸盘模式：0为自动模式，1为手动模式。"""
        if mode not in [0, 1]:
            raise ValueError("无效的模式。模式应为0（自动）或1（手动）。")
        return self._set_var(self.MOD, mode)

    def set_vacuum_levels(self, max_vacuum: int, min_vacuum: int):
        """设置最大和最小真空度。"""
        if not (0 <= max_vacuum <= 100):
            raise ValueError("最大真空度应在0到100%之间。")
        if not (0 <= min_vacuum <= max_vacuum):
            raise ValueError("最小真空度应在0到最大真空度之间。")
        print(f"设置最大真空度: {max_vacuum}% 和 最小真空度: {min_vacuum}%")
        self._set_var(self.MAX_VACUUM, 100 - max_vacuum)
        self._set_var(self.MIN_VACUUM, 100 - min_vacuum)

    def set_timeout(self, timeout: int):
        """设置超时时间。"""
        return self._set_var(self.TIMEOUT, timeout)

    def grip(self):
        """执行抓取操作。"""
        return self._set_var(self.GTO, 1)

    def release(self):
        """执行释放操作。"""
        return self._set_var(self.GTO, 0)

    def get_current_vacuum(self):
        """获取当前真空度。"""
        return 100 - self._get_var(self.CURRENT_VACUUM)

    def get_mode(self):
        """获取当前模式（自动或手动）。"""
        mode = self._get_var(self.MOD)
        return "自动" if mode == 0 else "手动"

    def get_object_status(self):
        """获取当前物体检测状态。"""
        status = self._get_var(self.OBJ)
        return "未检测到物体" if status == 3 else "检测到物体"

    def get_gripper_status(self):
        """获取电动吸盘的当前状态（激活或未激活）。"""
        status = self._get_var(self.STATUS)
        return "电动吸盘已激活" if status == 3 else "电动吸盘未激活"

    def get_max_vacuum(self):
        """获取最大真空度。"""
        return 100 - self._get_var(self.MAX_VACUUM)

    def get_min_vacuum(self):
        """获取最小真空度。"""
        return 100 - self._get_var(self.MIN_VACUUM)

    def _set_var(self, variable: str, value: int):
        """线程安全地发送命令来设置寄存器的值。"""
        with self.command_lock:
            command = f"SET {variable} {value}\n"
            try:
                self.socket.sendall(command.encode(self.ENCODING))
                response = self.socket.recv(1024)
                if not self._is_ack(response):
                    raise RuntimeError(f"设置{variable}为{value}失败。响应: {response.decode(self.ENCODING)}")
            except (socket.error, socket.timeout) as e:
                print(f"错误：设置{variable}时通信出错: {e}")
                raise
        return True

    def _get_var(self, variable: str):
        """获取寄存器的值。"""
        with self.command_lock:
            command = f"GET {variable}\n"
            try:
                self.socket.sendall(command.encode(self.ENCODING))
                response = self.socket.recv(1024)
                return self._parse_response(response, variable)
            except (socket.error, socket.timeout) as e:
                print(f"错误：获取{variable}时通信出错: {e}")
                raise

    def _parse_response(self, data: bytes, variable: str):
        """解析电动吸盘的响应并验证格式。"""
        try:
            decoded = data.decode(self.ENCODING)
            var_name, value_str = decoded.split()
            if var_name != variable:
                raise ValueError(f"意外响应: {decoded}")
            return int(value_str)
        except Exception as e:
            raise ValueError(f"无法解析响应: {data}。错误: {e}")

    @staticmethod
    def _is_ack(data: bytes):
        """检查响应是否为确认。"""
        return data == b'ack'

# 示例使用
if __name__ == "__main__":
    ip = '192.168.0.10'
    gripper = EPickGripper(ip)
    try:
        gripper.connect(ip, 63352)  # 替换为实际IP和端口
        print(f"物体状态: {gripper.get_object_status()}")
        gripper.set_mode(1)  # 设置为手动模式
        gripper.set_timeout(30)  # 设置超时时间为3秒
        time.sleep(0.2)
        gripper.set_vacuum_levels(75, 55)  # 设置最大真空度75%和最小真空度55%
        print(f"当前模式: {gripper.get_mode()}")
        print(f"最大真空度: {gripper.get_max_vacuum()}%")
        print(f"最小真空度: {gripper.get_min_vacuum()}%")
        print(f"电动吸盘状态: {gripper.get_gripper_status()}")
        # 执行抓取操作
        print("执行抓取操作...")
        gripper.grip()
        for _ in range(200):  # 每隔0.05秒输出当前真空度，共10秒
            time.sleep(0.05)
            print(f"当前真空度: {gripper.get_current_vacuum()}%")
        # 执行释放操作
        print("执行释放操作...")
        gripper.release()

    finally:
        gripper.disconnect()