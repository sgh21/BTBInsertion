from ultralytics import YOLO
import cv2
from  config import PLUG_MODEL_PATH,DATA_DIR

class Result:
    def __init__(self,**kwargs):
        self.img = getattr(kwargs, 'img', None)
        self.xyxy = getattr(kwargs, 'xyxy', None)
        self.boxes = getattr(kwargs, 'boxes', None)
        self.masks = getattr(kwargs, 'masks', None)
        self.classes = getattr(kwargs, 'classes', None)
    
    def result_from_yolo(self, yolo_result,img=None):
        self.img = img
        self.xyxy = yolo_result.boxes.xyxy
        self.boxes = yolo_result.boxes
        self.masks = yolo_result.masks.data
        self.classes = yolo_result.boxes.cls
        return self
    
    def result_filter_with_area(self,area_threshold=1e4):
        filtered_xyxy = []
        for i in range(len(self.xyxy)):
            box = [int(j.item()) for j in self.xyxy[i].cpu().numpy().flatten()]
            _area = (box[2] - box[0]) * (box[3] - box[1])
            if area_threshold <= _area:
                filtered_xyxy.append(box)
        self.xyxy = filtered_xyxy
        return self
    
    def result_box_padding(self,padding = 20):
        padding_box = []
        for j in range(len(self.xyxy)):
            box = self.xyxy[j]
            if box[0] - padding >= 0:
                box[0] -= padding
            if box[1] - padding >= 0:
                box[1] -= padding
            if box[2] + padding <= self.img.shape[1]:
                box[2] += padding
            if box[3] + padding <= self.img.shape[0]:
                box[3] += padding
            padding_box.append(box)
        self.xyxy = padding_box
        return self
    
class CutImage:
    def __init__(self, img, xyxy):
        self.img = img
        self.xyxy = xyxy
        
class YOLODetection:
    def __init__(self, model_path=PLUG_MODEL_PATH):
        self.model = YOLO(model_path, task='segment')
        self.result = Result()
        
    def load_image(self, img_path):
        """
        加载图像，返回cv.Image对象

        Args:
            img_path (str): 图像路径

        Returns:
            cv.Image: 图像对象
        """
        src = cv2.imread(img_path)
        return src
    
    def predict(self, img, padding = 20,area_threshold=1e2):
        """
        使用yolo模型进行预测，返回Result对象

        Args:
            img (cv.Image): input image
            padding (int, optional): padding the image to some pixel. Defaults to 20.
            area_threshold (float, optional): fliter the image that smaller than threshold. Defaults to 1e2.

        Returns:
            Result: the struct of result
        """
        result = self.model(img)[0]
        
        # 初始化1个Result对象
        self.result = self.result.result_from_yolo(result,img)
        # 过滤面积小于area_threshold的box
        self.result = self.result.result_filter_with_area(area_threshold)
        # 对box进行padding
        self.result = self.result.result_box_padding(padding)
        return self.result
    
    def cut_image(self):
        """
        将检测到的box进行裁剪，返回裁剪后的图像列表

        Returns:
            list[[cv.Image,xyxy]]: 图像列表，每个数据为一个裁剪后的图像
        """
        cut_img_list = []
        for xyxy in self.result.xyxy:
            x1, y1, x2, y2 = xyxy
            cutted_img = self.result.img[y1:y2, x1:x2]
            cut_img = CutImage(cutted_img, xyxy)
            cut_img_list.append(cut_img)
        
        return cut_img_list
            
    def show_result(self):
        for i in range(len(self.result.xyxy)):
            box = self.result.xyxy[i]
            cv2.rectangle(self.result.img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow('result', self.result.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_cut_image(self):
        cut_img_list = self.cut_image()
        for cut_img in cut_img_list:
            cv2.imshow('cut_img', cut_img.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
if __name__ == '__main__':
    yolo_detection = YOLODetection()
    img_path = f'{DATA_DIR}/test_image[0.5,0,0].png'
    img = yolo_detection.load_image(img_path)
    result = yolo_detection.predict(img)
    # print(result.xyxy)
    # yolo_detection.show_result()
    yolo_detection.show_cut_image()