import cv2
import torch
from ultralytics import YOLO
import os
import sys
sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}")
import numpy as np
from ultralytics.utils import ops

class YOLODetection:
    def __init__(self, model_path=f'{os.path.abspath(os.path.dirname(__file__))}/yolo_m.pt'):
        self.model = YOLO(model_path, task='segment')
        self.xyxy = None
        self.color_mask = None
        self.boxes = None
        self.classes = None

    def load_model(self, img_path):
        src = cv2.imread(img_path)
        return src

    def predict(self, img, area_min=10000, area_max=6000000):
        result = self.model(img)
        height, width = img.shape[:2]
        self.xyxy = result[0].boxes.xyxy
        self.classes = result[0].boxes.cls
        masks_data = result[0].masks.data
        self.color_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for index, mask in enumerate(masks_data):
            mask = mask.cpu().numpy() * 255
            mask = ops.scale_image(mask, img.shape)
            mask = mask.astype(np.uint8)
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.color_mask[mask_resized > 0] = [243, 98, 217]  # Green color mask
        self.boxes = []
        filtered_xyxy = []
        for i in range(len(self.xyxy)):
            box = [int(j.item()) for j in self.xyxy[i].cpu().numpy().flatten()]
            _area = (box[2] - box[0]) * (box[3] - box[1])
            if area_min <= _area <= area_max:
                filtered_xyxy.append(torch.tensor(box))
        for j in range(len(filtered_xyxy)):
            box = [int(k.item()) for k in filtered_xyxy[j].numpy().flatten()]
            if box[0] - 20 >= 0:
                box[0] -= 20
            if box[1] - 20 >= 0:
                box[1] -= 20
            if box[2] + 20 <= width:
                box[2] += 20
            if box[3] + 20 <= height:
                box[3] += 20
            self.boxes.append(box)
        return self.boxes, self.color_mask

    def draw_pca_direction(self, image, contour):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask[mask == 255] = 1
        points = cv2.findNonZero(mask)
        data_points = points[:, 0, :]
        mean = np.mean(data_points, axis=0)
        centered_data = data_points - mean
        cov_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        scale_factor = [100, 50]
        center = (mean[0], mean[1])
        main_direction = eigenvectors[:, 0]
        y_axis = np.array([0, 1])
        dot_product = np.dot(main_direction, y_axis)
        norm_product = np.linalg.norm(main_direction) * np.linalg.norm(y_axis)
        angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        cross_product = np.cross([main_direction[0], main_direction[1], 0], [0, 1, 0])[2]
        if cross_product < 0:
            angle_deg = - angle_deg
        for i in range(2):
            end_point = (
                int(mean[0] + scale_factor[i] * eigenvectors[0, i]),
                int(mean[1] + scale_factor[i] * eigenvectors[1, i])
            )
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.arrowedLine(image, (int(center[0]), int(center[1])), end_point, color, 1)
        return center, angle_deg

    def process_image(self, img, visualize=False):
        boxes, color_mask = self.predict(img)
        img_roi = img[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
        mask_roi = color_mask[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
        img_ready_copy = img_roi.copy()
        mask_roi = cv2.cvtColor(mask_roi, cv2.COLOR_BGR2GRAY)
        _, mask_roi = cv2.threshold(mask_roi, 127, 255, cv2.THRESH_BINARY)
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_roi = cv2.dilate(mask_roi, kernal, iterations=1)
        imgResult_socket = cv2.bitwise_and(img_roi, img_roi, mask=mask_roi)
        white_background = np.ones_like(img_roi) * 255
        imgResult_socket[~mask_roi.astype(bool)] = white_background[~mask_roi.astype(bool)]
        img_ready = cv2.GaussianBlur(imgResult_socket, (7, 7), 0)
        img_ready_lab = cv2.cvtColor(img_ready, cv2.COLOR_BGR2LAB)
        lower = np.array([0, 0, 0])
        upper = np.array([75, 130, 130])
        mask_socket = cv2.inRange(img_ready_lab, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphologyEx = cv2.morphologyEx(mask_socket, cv2.MORPH_OPEN, kernel, iterations=1)
        morphologyEx = cv2.morphologyEx(morphologyEx, cv2.MORPH_CLOSE, kernel, iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        morphologyEx = cv2.erode(morphologyEx, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(morphologyEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                max_contour = contour
        if max_contour is not None:
            cv2.drawContours(img_ready_copy, [max_contour], -1, (0, 255, 0), 2)
            center, angle_deg = self.draw_pca_direction(img_ready_copy, max_contour)
            cX, cY = boxes[0][0] + int(center[0]), boxes[0][1] + int(center[1])
            center = (boxes[0][0] + center[0], boxes[0][1] + center[1])
            cv2.circle(img_ready_copy, (int(cX), int(cY)), 1, (255, 0, 0), -1)
            cv2.putText(img_ready_copy, "Center", (int(cX) - 20, int(cY) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(img, (cX, cY), 1, (255, 0, 0), -1)
        if visualize:
            self.visualize(img, img_roi, mask_roi, imgResult_socket, mask_socket, morphologyEx, img_ready_copy)
        return center, angle_deg

    def visualize(self, img, img_roi, mask_roi, imgResult_socket, mask_socket, morphologyEx, img_ready_copy):
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', mask_roi.shape[1], mask_roi.shape[0])
        cv2.imshow('result', img_roi)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mask', mask_roi.shape[1], mask_roi.shape[0])
        cv2.imshow('mask', mask_roi)
        cv2.namedWindow('imgResult_socket', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imgResult_socket', mask_roi.shape[1], mask_roi.shape[0])
        cv2.imshow('imgResult_socket', imgResult_socket)
        cv2.namedWindow('mask_socket', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mask_socket', mask_roi.shape[1], mask_roi.shape[0])
        cv2.imshow('mask_socket', mask_socket)
        cv2.namedWindow('morphologyEx', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('morphologyEx', mask_roi.shape[1], mask_roi.shape[0])
        cv2.imshow('morphologyEx', morphologyEx)
        cv2.namedWindow('img_ready_copy', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img_ready_copy', mask_roi.shape[1], mask_roi.shape[0])
        cv2.imshow('img_ready_copy', img_ready_copy)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', img.shape[1], img.shape[0])
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    yolo = YOLODetection()
    # template_center = (1225.0030, 1016.3557)
    # template_angle = -2.1577
    template_center = (1223.22738, 1067.66969)
    template_angle = -1.87683
    img_path = '../result/hk_camera/hk_image_24p035.png'
    img = yolo.load_model(img_path)
    center, angle = yolo.process_image(img, visualize=True)
    print(f"Center: {center}")
    print(f"Angle: {angle}")
    print("delta_x:", (center[0] - template_center[0])*0.0212)
    print("delta_y:", (center[1] - template_center[1])*0.0212)
    print("delta_angle:", -(angle - template_angle))