from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import time
import cv2
import tempfile
import os
import uuid

# Cần điều chỉnh đường dẫn import tuỳ thuộc vào cách cấu trúc  
from app.core.model import FireDetector
from app.core.mailer import mailer

app = FastAPI(
    title="Fire Detection API",
    description="API nhận diện hình ảnh/video có chứa đám cháy bằng YOLOv5.",
    version="1.0.0"
)

# 1. Cấu hình CORS để web frontend gọi được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_sessions = {}

# 2. Khởi tạo Neural Network Model
print("Đang khởi động Model YOLOv5...")
try:
    detector = FireDetector(weights_path='model/yolov5s_best.pt')
except Exception as e:
    print(f"LỖI LOAD MODEL: {e}")

# 3. Kết nối thư mục `static` để hiển thị UI Web
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/detect/image")
async def detect_api(file: UploadFile = File(...), email: str = Form(None)):
    """API dùng để nhận diện đám cháy cho LUỒNG ẢNH & WEBCAM."""
    
    if file.filename and file.filename.endswith(('.mp4', '.avi', '.mov')):
        return Response(content="Lỗi: Hãy sang tab Xử lý Video!", status_code=400)
    
    start_time = time.time()
    content = await file.read()
    
    # Gửi vào YOLOv5 chạy
    try:
        image_bytes_out, metrics = detector.detect_image(content)
    except Exception as e:
        return Response(content=f"Lỗi đọc file ảnh: {str(e)}", status_code=400)
    
    end_time = time.time()
    inference_ms = round((end_time - start_time) * 1000)
    
    response = Response(content=image_bytes_out, media_type="image/jpeg")
    
    if len(metrics) > 0:
        max_conf = max([d['confidence'] for d in metrics]) * 100
        response.headers["X-Fire-Detected"] = "true"
        response.headers["X-Highest-Confidence"] = f"{round(max_conf, 1)}"
        
        # GỬI EMAIL THÔNG BÁO
        if email and email.strip() != "":
            mailer.send_alert_async(email, image_bytes_out, max_conf)
    else:
        response.headers["X-Fire-Detected"] = "false"
        
    response.headers["X-Inference-Time-Ms"] = str(inference_ms)
    return response

# --- MODULE VIDEO ---
def generate_video_frames(video_path: str, user_email: str = None):
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.resize(frame, (640, 480))
            out_frame, metrics = detector.detect_frame(frame)
            
            if len(metrics) > 0 and user_email:
                max_conf = max([d['confidence'] for d in metrics]) * 100
                if max_conf > 50.0:
                    ret_m, buffer_mail = cv2.imencode('.jpg', out_frame)
                    if ret_m:
                        mailer.send_alert_async(user_email, buffer_mail.tobytes(), max_conf)
            
            ret, buffer = cv2.imencode('.jpg', out_frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

@app.post("/api/detect/video/upload")
async def upload_video_api(file: UploadFile = File(...), email: str = Form(None)):
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, 'wb') as f:
        f.write(await file.read())
        
    vid_id = str(uuid.uuid4())
    video_sessions[vid_id] = {
        "path": temp_path,
        "email": email
    }
    return {"video_id": vid_id}

@app.get("/api/detect/video/stream/{video_id}")
async def stream_video_api(video_id: str):
    session = video_sessions.get(video_id)
    if not session:
        return Response("Invalid Stream ID or Stream Ended", status_code=404)
        
    path = session["path"]
    email = session["email"]
    
    # Chỉ stream một lần
    del video_sessions[video_id]
    
    return StreamingResponse(
        generate_video_frames(path, email),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/")
def read_root():
    return {"message": "Server đang chạy."}

@app.get("/favicon.ico")
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="#ff4757" d="M12 2C12 2 15 6 15 9.5C15 11.2 13.7 12.5 12 12.5C10.3 12.5 9 11.2 9 9.5C9 6 12 2 12 2ZM17.5 12.5C17.5 16.1 14.6 19 11 19C7.4 19 4.5 16.1 4.5 12.5C4.5 10.9 5.1 9.4 6 8.3C6 8.3 4 10.5 4 13C4 17.4 7.6 21 12 21C16.4 21 20 17.4 20 13C20 9 17.5 6 17.5 6C17.5 6 17.5 8.9 17.5 12.5Z"/></svg>'
    return Response(content=svg, media_type="image/svg+xml")

