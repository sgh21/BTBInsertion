from utils.yolo_dection import YOLODetection
from config import PARAMS
import cv2
import numpy as np

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """保持宽高比的图像缩放"""
    dim = None
    (h, w) = image.shape[:2]
    
    # 由于图像尺寸太小，在process_cv直接进行腐蚀和膨胀导致大量信息丢失，
    # 请你帮我先等比例缩放到大尺寸，进行形态学操作，主成分分析，再最后返回到正常的比例
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def yolo_detection():
    model_path = PARAMS['yolo_model_path']
    data_dir = PARAMS['data_dir']
    test_img = f'{data_dir}/test_image[0,0,0].png'
    yolo = YOLODetection(model_path)
    img = yolo.load_image(test_img)
    yolo_result = yolo.predict(img)
    cut_img_list = yolo.cut_image()
    
    return yolo_result, cut_img_list
def scale_img(img, scale_factor=2.0, interpolation=cv2.INTER_LINEAR):
    # 记录原始尺寸
    original_height, original_width = img.shape[:2]
    
    # 先将图像放大
    scaled_width = int(original_width * scale_factor)
    scaled_height = int(original_height * scale_factor)
    img_scaled = cv2.resize(img, (scaled_width, scaled_height), interpolation=interpolation)
    
    return img_scaled

def process_cv(cut_img_list, scale_factor=2.0):
    """
    处理裁剪后的图像
    :param cut_img_list: 裁剪图像列表，每个元素包含img和xyxy属性
    :param scale_factor: 缩放因子，用于放大和缩小图像
    :return: 处理结果列表，每个元素包含图像处理中间结果和最终结果
    """
    all_results = []
    for cut_img in cut_img_list:
        img_roi = cut_img.img.copy()
        box = cut_img.xyxy
        
        # 记录原始尺寸
        original_height, original_width = img_roi.shape[:2]
        
        # 先将图像放大
        img_scaled = scale_img(img_roi, scale_factor=scale_factor)
        
        # 转换到HSV颜色空间
        img_hsv = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2HSV)
        
        # 提取黑色区域
        lower = np.array(PARAMS['hsv_lower_bound'])
        upper = np.array(PARAMS['hsv_upper_bound'])
        mask = cv2.inRange(img_hsv, lower, upper)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 先腐蚀再膨胀，去掉毛刺
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 先膨胀再腐蚀，填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 去除杂质色块
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [max_contour], -1, 255, -1)
        mask = cv2.bitwise_and(mask, contour_mask)
        
        # 计算PCA
        points = cv2.findNonZero(mask)
        data_points = points[:, 0, :]
        mean = np.mean(data_points, axis=0)
        
        # 计算主方向
        centered_data = data_points - mean
        cov_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 计算角度
        main_direction = eigenvectors[:, 0]
        y_axis = np.array([0, 1])
        dot_product = np.dot(main_direction, y_axis)
        norm_product = np.linalg.norm(main_direction) * np.linalg.norm(y_axis)
        angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # 确定角度方向
        cross_product = np.cross([main_direction[0], main_direction[1], 0], [0, 1, 0])[2]
        if cross_product < 0:
            angle_deg = -angle_deg
        
        # 缩小回原始尺寸
        img_final = cv2.resize(img_scaled, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        mask_final = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        max_contour_scaled = (max_contour / scale_factor).astype(int)
        mean_scaled = (mean / scale_factor)
        main_direction_scaled = main_direction  # 方向不变
        
        # 计算全局坐标系中的中心点
        center = (box[0] + mean_scaled[0], box[1] + mean_scaled[1])
        
        # 准备可视化结果
        result_img = img_roi.copy()
        cv2.drawContours(result_img, [max_contour_scaled], -1, (0, 255, 0), 1)
        cv2.circle(result_img, (int(mean_scaled[0]), int(mean_scaled[1])), 2, (255, 0, 0), -1)
        cv2.line(result_img, 
                (int(mean_scaled[0]), int(mean_scaled[1])), 
                (int(mean_scaled[0] + main_direction_scaled[0] * 50 / scale_factor), 
                 int(mean_scaled[1] + main_direction_scaled[1] * 50 / scale_factor)), 
                (0, 0, 255), 2)
                
        all_results.append({
            'center': center,
            'angle': angle_deg,
            'visualization': {
                'original': img_roi,
                'hsv': cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR),
                'mask': mask_final,
                'result': result_img
            }
        })
        
    return all_results

def visualize_result(result, scale_width=800,save_path=None):
    
    """可视化处理结果"""
    # 获取所有图像
    img_roi = result['visualization']['original']
    img_hsv = result['visualization']['hsv']
    mask = result['visualization']['mask']
    result_img = result['visualization']['result']
    angle_deg = result['angle']
    
    # 将mask转换为3通道图像
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 缩放所有图像
    img_roi_resized = resize_with_aspect_ratio(img_roi, width=scale_width)
    img_hsv_resized = resize_with_aspect_ratio(img_hsv, width=scale_width)
    mask_resized = resize_with_aspect_ratio(mask_color, width=scale_width)
    result_img_resized = resize_with_aspect_ratio(result_img, width=scale_width)
    
    # 创建可视化布局
    vis_row1 = np.hstack([img_roi_resized, img_hsv_resized])
    vis_row2 = np.hstack([mask_resized, result_img_resized])
    visualization = np.vstack([vis_row1, vis_row2])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img_roi_resized.shape[:2]
    
    cv2.putText(visualization, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(visualization, 'HSV', (w + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(visualization, 'Mask', (10, h + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(visualization, f'Result (Angle: {angle_deg:.1f})', 
                (w + 10, h + 30), font, 1, (0, 255, 0), 2)
    
    cv2.imshow('Vision Process', visualization)
    if save_path:
        cv2.imwrite(save_path, visualization)
    key = cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return key == ord('q')

def main():
    yolo_result, cut_img_list = yolo_detection()
    results = process_cv(cut_img_list, scale_factor=4.0)  # 你可以调整缩放因子
    
    for i, result in enumerate(results):
        print(f"Object {i}:")
        print(f"Center: {result['center']}")
        print(f"Angle: {result['angle']}")
        
        if visualize_result(result, scale_width=600):
            break

if __name__ == '__main__':
    main()
