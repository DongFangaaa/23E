import threading
import time
import numpy as np

class YOLO_Work:
    def __init__(self, model, pad=30):
        self.model = model
        self.pad = pad # 取更多区块 防止太近而取少
        self.lock = threading.Lock()
        self.latest_frame = None  # 接收前台喂进来的新图
        self.current_roi = None   # 后台算出的最新小框界限
        self.running = True
        
        # 开局自动启动独立后台线程
        self.thread = threading.Thread(target=self.work_loop, daemon=True)
        self.thread.start()

    def work_loop(self):
        """
        后台无限循环函数 把前台的图算完更新给 current_roi
        """
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                try:
                    # 在独立内核跑256分辨率推理
                    results = self.model(frame_to_process, imgsz=256, verbose=False, conf=0.7)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        length, width = frame_to_process.shape[:2]
                        box = results[0].boxes[0].xyxy.cpu().numpy().squeeze()
                        
                        x1 = max(0, int(box[0]) - self.pad)
                        y1 = max(0, int(box[1]) - self.pad)
                        x2 = min(width, int(box[2]) + self.pad)
                        y2 = min(length, int(box[3]) + self.pad)
                        
                        with self.lock:
                            self.current_roi = (x1, y1, x2, y2)
                except Exception:
                    pass
            else:
                time.sleep(0.002)

    def feed_and_get_roi(self, frame, fallback_clear=False):
        """
        前台调用函数 把当前帧喂给后台并拿出当前最新的ROI坐标
        """
        with self.lock:
            self.latest_frame = frame
            if fallback_clear:
                self.current_roi = None
            return self.current_roi

    def update_roi_from_otsu(self, new_roi):
        """
        前台大津法更新吸附坐标 双方协同校正
        """
        with self.lock:
            self.current_roi = new_roi

    def close(self):
        self.running = False