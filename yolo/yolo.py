import cv2
import numpy as np
from ultralytics import YOLO
import sys

sys.path.append(r'D:\Git\23年E题\视觉 串口')

from uart_driver import UART_Sender
from pnp_solve import PnP_solve

# ========================================
# 第一部分：全局初始化
# ========================================
model = YOLO('best.pt')            # 你自己训练的靶纸/目标模型
cap = cv2.VideoCapture(1)
uart = UART_Sender(port='COM18', baudrate=115200)
pnp = PnP_solve(rect_width=215.0, rect_length=305.0, focal_length=600.0)

# ========================================
# 第二部分：主循环
# ========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- 阶段 A：YOLO 目标检测 ----------
    results = model(frame, conf=0.5, verbose=False)
    result = results[0]

    found = False
    error_x, error_y = 0.0, 0.0
    dist_z = 0.0

    if len(result.boxes) > 0:
        found = True
        box = result.boxes[0]      # 取置信度最高的目标

        # 提取中心坐标
        x, y, w, h = box.xywh[0].tolist()

        # 提取边界框角点（给 PnP 用）
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        corners = np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype="float32")

        # ---------- 阶段 B：PnP 姿态解算 ----------
        success, dist_z, offset_x, offset_y, *_ , rvec, tvec = pnp.solve(corners)

        if success:
            error_x = offset_x
            error_y = offset_y
            pnp.draw_axes(frame, rvec, tvec, axes_length=100.0)

    # ---------- 阶段 C：串口通信 ----------
    if found:
        uart.send_error(error_x, error_y, 0, 1, dist_z)
    else:
        uart.target_lost()

    # ---------- 阶段 D：画面显示 ----------
    annotated = result.plot()
    cv2.putText(annotated, f"err:({error_x:.1f},{error_y:.1f}) dist:{dist_z:.0f}mm",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 E-Competition", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========================================
# 第三部分：资源释放
# ========================================
cap.release()
cv2.destroyAllWindows()
uart.close()