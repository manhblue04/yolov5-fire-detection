import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import threading

class AlertMailer:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        self.sender_email = "manhducnb204@gmail.com"
        self.sender_password = "hbds fudo eotr zikv" 
        
        self.last_sent_times = {}
        self.COOLDOWN_SECONDS = 3 * 60 
        
        self._lock = threading.Lock()

    def set_credentials(self, email, password):
        self.sender_email = email
        self.sender_password = password

    def can_send(self, to_email: str) -> bool:
        """Kiểm tra xem đã qua 3 phút kể từ lần gửi cuối cho email này chưa"""
        with self._lock:
            now = time.time()
            last_sent = self.last_sent_times.get(to_email, 0)
            if now - last_sent >= self.COOLDOWN_SECONDS:
                return True
            return False

    def mark_sent(self, to_email: str):
        with self._lock:
            self.last_sent_times[to_email] = time.time()

    def send_alert_async(self, to_email: str, image_bytes: bytes, confidence: float):
        """Gửi email không đồng bộ để không đứng hệ thống"""
        if not to_email or "@" not in to_email:
            return False
            
        if not self.can_send(to_email):
            print(f"BỎ QUA GỬI MAIL tới {to_email}: Chưa hết thời gian chờ 3 phút.")
            return False

        # Khởi chạy một thread gửi mail
        thread = threading.Thread(target=self._send_email, args=(to_email, image_bytes, confidence))
        thread.daemon = True
        thread.start()
        
        # Đánh dấu thời gian đã gửi
        self.mark_sent(to_email)
        return True

    def _send_email(self, to_email: str, image_bytes: bytes, confidence: float):
        print(f"Đang gửi Email Cảnh báo Tới: {to_email}...")
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = f"🚨 CẢNH BÁO CHÁY: Hệ thống FireGuardian phát hiện nguy hiểm ({round(confidence)}%)!"

            # Phần text
            body = f"""
            <h2>Kích hoạt cảnh báo cháy rủi ro cao!</h2>
            <p>Hệ thống Trí tuệ Nhân tạo YOLOv5 đã phát hiện dấu hiệu hỏa hoạn với độ tin cậy là <strong>{round(confidence, 1)}%</strong>.</p>
            <p>Mời bạn xem hình ảnh chụp được đính kèm bên dưới để xác nhận.</p>
            <br>
            <i>Thông báo này được gửi tự động. Vui lòng kiểm tra khẩn cấp!</i>
            """
            msg.attach(MIMEText(body, 'html'))

            # Phần ảnh đính kèm
            if image_bytes:
                image = MIMEImage(image_bytes, name="fire_alert.jpg")
                msg.attach(image)

            # Kết nối server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            # Bỏ qua đăng nhập nếu chưa thiết lập mật khẩu thật (tránh crash)
            if self.sender_password:
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                print(f"Gửi mail Tới {to_email} THÀNH CÔNG!")
            else:
                print("Chưa thiết lập Mật khẩu Người Gửi (App Password). Bỏ qua gửi thực tế.")
                
            server.quit()
        except Exception as e:
            print(f"LỖI GỬI EMAIL: {e}")
            # Nếu lỗi, xóa block thời gian để cho gửi lại ở lần sau ngay lập tức
            self.last_sent_times[to_email] = 0

# Khởi tạo instance toàn cục (Singleton)
mailer = AlertMailer()
