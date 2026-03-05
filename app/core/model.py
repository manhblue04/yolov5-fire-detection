import torch
import cv2
import numpy as np
import io
from PIL import Image

class FireDetector:
    def __init__(self, weights_path='../model/yolov5s_best.pt'):
        # Tải mô hình YOLOv5 từ Ultralytics (thường yêu cầu clone repo yolov5)
        # Vì ta đã có repo yolov5 ở thư mục ../yolov5 nên load local từ đó
        print(f"Loading model from {weights_path}...")
        self.model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')
        # Tối ưu hóa cho quá trình suy luận
        self.model.eval()
        self.model.conf = 0.25  # Ngưỡng tin cậy (Confidence threshold)
        self.model.iou = 0.45   # NMS IoU threshold

    def detect_image(self, file_bytes):
        # 1. Chuyển byte ảnh từ API nhận được thành dạng mảng OpenCV Numpy
        image_stream = io.BytesIO(file_bytes)
        img = Image.open(image_stream).convert('RGB')
        img_np = np.array(img)
        
        # 2. Chạy model trực tiếp bằng Ultralytics YOLOv5 
        results = self.model(img_np)
        
        # 3. Lấy ảnh đầu ra đã được vẽ sẵn khung chữ nhật bao quanh đám cháy (Bounding Box)
        # Kết quả được numpy render tự động
        rendered_images = results.render()
        img_with_boxes = rendered_images[0]
        
        # 4. Trích xuất thông tin phát hiện (số lượng đám cháy, độ tin cậy)
        df = results.pandas().xyxy[0]
        detections = []
        for index, row in df.iterrows():
            detections.append({
                "class": int(row['class']),
                "name": row['name'],
                "confidence": float(row['confidence']),
                "box": [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
            })
            
        # 5. Encode lại dạng byte jpeg để gửi về frontend
        # Open CV dùng BGR thay vì RGB nên cần convert lại
        img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode('.jpg', img_bgr)
        return encoded_image.tobytes(), detections

    def detect_frame(self, frame_bgr):
        """Xử lý trực tiếp 1 khung hình (OpenCV BGR) - Phục vụ luồng Video/Webcam"""
        # Chuyển BGR (OpenCV) sang RGB (YOLOv5)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Suy luận
        results = self.model(img_rgb)
        
        # Render Box
        rendered_images = results.render()
        img_with_boxes = rendered_images[0]
        
        # Trích xuất thông tin phát hiện
        df = results.pandas().xyxy[0]
        detections = []
        for index, row in df.iterrows():
            detections.append({
                "class": int(row['class']),
                "name": row['name'],
                "confidence": float(row['confidence']),
                "box": [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
            })
            
        # Convert lại BGR để hiển thị OpenCV
        img_bgr_out = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        return img_bgr_out, detections
