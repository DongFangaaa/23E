import numpy as np
import cv2
import json

class Laser_Calibrator:
    def __init__(self):
        self.P_cam_1 = None
        self.P_cam_2 = None

    def capture_calibration_point(self, board_mm_x, board_mm_y, rvec, tvec):
        """
        采样当前距离下的真实三维射线穿透点
        """
        R, _ = cv2.Rodrigues(rvec)
        board_point = np.array([[board_mm_x], [board_mm_y], [0.0]], dtype=np.float64)
        # 正向转换至相机镜头 3D 空间
        P_cam = R @ board_point + tvec.reshape(3, 1)
        return P_cam #得到在摄像机上的空间坐标

    def calibrate(self, P1, P2):
        """
        传入近距离采样点 P1 与远距离采样点 P2，解出绝对起点 O_cam 与斜向 D_cam
        """
        diff = P2 - P1 #向量
        dist = np.linalg.norm(diff) #模长
        if dist < 100.0:
            print("距离太近了")
            return None, None

        # 解出精确倾斜方向
        D_cam = diff / dist #偏移的方向向量

        # 反向倒推回镜头表面平面 Z=0
        lam = P1[2, 0] / D_cam[2, 0]
        O_cam = P1 - lam * D_cam #摄像头面上的激光笔位置

        print("\n" + "="*50)
        print(f"O_cam: X={O_cam[0,0]:.16f}mm,\n Y={O_cam[1,0]:.16f}mm,\n Z={O_cam[2,0]:.1f}mm\n")
        print(f"D_cam: X={D_cam[0,0]:.16f},\n Y={D_cam[1,0]:.16f},\n Z={D_cam[2,0]:.16f}\n")
        print("="*50 + "\n")

        return O_cam.flatten(), D_cam.flatten()

if __name__ == "__main__":
    from pnp_solve import PnP_solve
    from vision import TargetDetector, LaserDetector

    #简易的两点求解
    print("\n"+"="*60)
    print("1键 -> 抓取 P1")
    print("2键 -> 抓取 P2")
    print("c键 -> 计算结果并存入 JSON")
    print("q键 -> 退出")
    print("="*60 + "\n")

    # 初始化
    point_solve = PnP_solve()
    target_detector = TargetDetector(point_solve)
    laser_detector = LaserDetector(point_solve)
    calibrator = Laser_Calibrator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头无法打开")
        exit(1)

    P1 = None
    P2 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取画面失败")
            continue

        # 视觉识别：抓靶心外框 + 抓激光点
        target_data = target_detector.detect(frame)
        laser_found, bx, by = laser_detector.detect(frame,target_data)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            if target_data['found'] and laser_found:
                rvec, tvec = target_data['rvec'], target_data['tvec']
                P1 = calibrator.capture_calibration_point(bx, by, rvec, tvec) #摄像机上的空间坐标
                print(f"p1 \n{P1.flatten()}\n") 
            else:
                print("P1没有识别到")

        elif key == ord('2'):
            if target_data['found'] and laser_found:
                rvec, tvec = target_data['rvec'], target_data['tvec']
                P2= calibrator.capture_calibration_point(bx, by, rvec, tvec)
                print(f"P2 \n{P2.flatten()}\n")
            else:
                print("P2没有识别到")

        elif key == ord('c'):
            if P1 is not None and P2 is not None:
                O_cam, D_cam = calibrator.calibrate(P1, P2)
                if O_cam is not None and D_cam is not None:
                    # 将计算出来的结果存入 JSON 文件
                    res_dict = {
                        "origin": O_cam.tolist(),
                        "direction": D_cam.tolist()
                    }
                    with open("laser_calib.json", "w", encoding="utf-8") as f:
                        json.dump(res_dict, f, indent=4)
                    print("偏置与斜角参数已存入当前目录下的：laser_calib.json")
            else:
                print("两个点未采全")

        elif key == ord('q'):
            print("退出自动标定程序。")
            break

        cv2.imshow("Laser Auto-Calibration Live Dashboard", frame)

    cap.release()
    cv2.destroyAllWindows()
