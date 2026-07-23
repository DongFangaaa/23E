"""
偏差以utf-8  Serror_x,error_y,self.target_state,self.state,dist_zE\n的形式发送
self.target_state 为0关闭激光 为1开启激光
self.state 为0停止运动
dist_z 为当前距离面板的距离
距离为毫米 float
须根据当前的距离来同步调整P的大小
dist_z越小P越大
error_x为正代表需要向左  为负代表需要向右  error_y为正代表需要向上 为负代表需要向下
注意 和之前的逻辑相反
例如 S30，50，0，1，150.0E\n  代表向左偏差为30 向上偏差50 激光关闭 云台运动 距离面板150.0毫米
目前有仅一种运动状态 状态1为回到中心
须传参 self.state 可为0 1 2 代表具体的运动状态
会同步返回self.state 若为0 则停止运动
注 在寻框失败时 暂定dist_z为0.5
注 当前为试验 实际运动时须改动237
"""

import os
import cv2
import numpy as np
from uart_driver import UART_Sender
from pnp_solve import PnP_solve
from maix import camera, display, image, nn, app

# 定义一个空的回调函数，用于滑动条（Trackbar）的回调参数
def nothing(x):
    pass

#顺序排列四个顶点 左上开始顺时针
def order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    pts = pts.reshape(4,2) #初始化

    s = pts.sum(axis=1) #单个xy算总
    rect[0] = pts[np.argmin(s)] #左上  np.argmin()得到的为索引
    rect[2] = pts[np.argmax(s)] #右下

    diff = np.diff(pts,axis=1) #单个y-x
    rect[1] = pts[np.argmin(diff)] #右上
    rect[3] = pts[np.argmax(diff)] #左下

    return rect

class TargetDetector:
    """
    先用训练好的YOLO找框 对寻找到的部分图像就行大津法
    寻找内外方框，计算中心坐标与多边形角点
    解算PnP姿态
    """
    def __init__(self,pnpsolve:PnP_solve):
        self.pnp = pnpsolve
        self.keeptime = 0
        self.kernel = np.ones((5,5),dtype=np.uint8)

    def detect(self,frame,yolo_objs):
        length , width = frame.shape[:2]
        use_roi = False
        roi_x1, roi_y1 = 0,0
        roi_x2, roi_y2 = width, length
        
        if len(yolo_objs) > 0:
            obj = yolo_objs[0]  # 外框的pos
            roi_x1 = max(0, obj.x - 45)
            roi_y1 = max(0, obj.y - 45)
            roi_x2 = min(width, obj.x + obj.w + 45)
            roi_y2 = min(length, obj.y + obj.h + 45)
            use_roi = True
            

        # 真正切下需要检测的图像  如果YOLO找到了框 则为找到的图 没找到则为原图
        detect_img = frame[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]

        gray_img = cv2.cvtColor(detect_img,cv2.COLOR_BGR2GRAY) # 转化为灰度图
        gray_Biur_img = cv2.GaussianBlur(gray_img,(5,5),0)

        #二值化
        """
        yolo找到黑框采用大津法 没找到则使用局部自适应 
        """
        if use_roi:
            _ , binary = cv2.threshold(gray_Biur_img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(gray_Biur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize=9,C=2)


        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel,iterations=2) #闭运算去除微小噪点
        #closed = cv2.dilate(closed, self.kernel, iterations=1)
        contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #找出轮廓

        valid_rects = [] #收集最后内外框

        if hierarchy is not None:
            hierarchy = hierarchy[0]

            for index, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if area <= 800:
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(
                    contour,
                    0.04 * perimeter,
                    True
                )

                if len(approx) != 4:
                    continue


                valid_rects.append({
                    "area": area,
                    "approx": approx,
                    "index": index,
                    "child": int(hierarchy[index][2]),
                    "parent": int(hierarchy[index][3]),
                })

        # 按面积从大到小排序。
        valid_rects.sort(key=lambda rect: rect["area"], reverse=True)

        def is_descendant(child_index, ancestor_index):
            """判断一个轮廓是否位于另一个轮廓的内部层级。"""
            parent_index = int(hierarchy[child_index][3])
            while parent_index != -1:
                if parent_index == ancestor_index:
                    return True
                parent_index = int(hierarchy[parent_index][3])
            return False

        outer_rect = None
        inner_rect = None

        # 优先选择具有真实父子层级、且中心接近的内外矩形。
        if hierarchy is not None:
            for outer_candidate in valid_rects:
                outer_center = np.mean(
                    outer_candidate["approx"].reshape(4, 2), axis=0
                ) #计算中心
                descendants = []

                for inner_candidate in valid_rects:
                    if inner_candidate["area"] >= outer_candidate["area"]: #面积嵌套
                        continue
                    if not is_descendant(inner_candidate["index"], outer_candidate["index"]):#轮廓嵌套
                        continue

                    inner_center = np.mean(
                        inner_candidate["approx"].reshape(4, 2), axis=0
                    )
                    if np.linalg.norm(outer_center - inner_center) < 50.0:
                        descendants.append(inner_candidate)

                if descendants:
                    outer_rect = outer_candidate
                    inner_rect = max(
                        descendants, key=lambda rect: rect["area"] #取面积最大的子轮廓
                    )
                    break

        # 噪声造成层级断裂时，退回原来的面积和中心距离判断。
        if outer_rect is None and len(valid_rects) >= 2:
            for outer_pos, outer_candidate in enumerate(valid_rects[:-1]):
                outer_center = np.mean(
                    outer_candidate["approx"].reshape(4, 2), axis=0
                )
                for inner_candidate in valid_rects[outer_pos + 1:]:
                    inner_center = np.mean(
                        inner_candidate["approx"].reshape(4, 2), axis=0
                    )
                    if np.linalg.norm(outer_center - inner_center) < 100.0:
                        outer_rect = outer_candidate
                        inner_rect = inner_candidate
                        break
                if outer_rect is not None:
                    break
                
        result = {
            'found': False,
            'center_x': int(width / 2), 'center_y': int(length / 2),
            'center_line':None,
            'dist_z': 0.0,'offset_x':0.0,'offset_y':0.0,
            'yaw': 0.0, 'pitch': 0.0,'roll':0.0,
            'rvec': None, 'tvec': None,
            'binary':binary,
        }

        if outer_rect is not None and inner_rect is not None:
            """
            把 ROI 里得到的顶角坐标加回偏移量，映射为全景原图坐标
            """
            roi_offset = np.array(
                [int(roi_x1), int(roi_y1)], dtype=np.int32
            )
            outer_approx = (
                outer_rect["approx"].reshape(4, 2) + roi_offset
            )
            inner_approx = (
                inner_rect["approx"].reshape(4, 2) + roi_offset
            )

            #内外框顶点排序
            ordered_outer = order_points(outer_approx)
            ordered_inner = order_points(inner_approx)

            #中线
            center_line = (ordered_outer + ordered_inner) / 2.0
            center_line = center_line.astype("int32")#中线顶点坐标  此时为浮点数

            center_x = int(np.mean(center_line[:,0])) #中线中心点x坐标
            center_y = int(np.mean(center_line[:,1])) #中线中心点y坐标

            #cv2.drawContours(frame,[outer_approx],-1,(255,0,0),1) #外框
            #cv2.drawContours(frame,[inner_approx],-1,(0,0,255),1) #内框
            #cv2.drawContours(frame,[center_line],-1,(0,255,0),3) #中线
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)#中心点

            #pnp姿态解算
            success, dist_z, offset_x, offset_y, yaw, pitch, roll, rvec, tvec = self.pnp.solve(ordered_outer)
            #if success:
                #self.pnp.draw_axes(frame,rvec,tvec,axes_length=100.0)

            self.keep_time = cv2.getTickCount()

            result.update({
            'found': True,
            'center_x': center_x, 'center_y': center_y,
            'center_line':center_line,
            'dist_z': dist_z if success else 0.0,
            'offset_x':offset_x if success else 0.0 ,
            'offset_y':offset_y if success else 0.0,
            'yaw': yaw if success else 0.0, 
            'pitch': pitch if success else 0.0,
            'roll': roll if success else 0.0,
            'rvec': rvec, 'tvec': tvec,
            'binary':binary,
            })

        return result
    
class LaserDetector:
    """
    从原图像中通过HSV色相定位蓝紫色激光
    并得到在pnp中的做标
    """
    def __init__(self,pnpsolve:PnP_solve):
        self.pnp = pnpsolve
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.first_point = True
        self.kernel = np.ones((5, 5), dtype=np.uint8)

    def detect(self,frame, clean_frame, result):
        board_found = False
        blurred = cv2.GaussianBlur(clean_frame,(3,3),0) #高斯模糊去噪
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV) # 转化为HSV
        
        # 蓝紫色激光
        lower = np.array([100, 200, 200], dtype=np.uint8)
        upper = np.array([160, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # mask = cv2.inRange(hsv, lower, upper)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 开运算去细白颗粒
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) # 闭运算填补可能的空缺

        mask = cv2.dilate(mask, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found = False
        px, py = 0, 0
        bx, by = 0.0, 0.0
        
        for contour in contours:
            if cv2.contourArea(contour) > 10: # 确认发光斑
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    found = True
                    cv2.drawMarker(frame, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                    break
                    
        # 换算至pnp
        if found and result['found'] and result['rvec'] is not None:
            bx , by = self.pnp.point_board(px, py, result['rvec'], result['tvec'])
            if bx is not None and by is not None:
                board_found = True
                cv2.putText(frame, f"Laser_World: ({bx:.1f}, {by:.1f})mm", \
                            (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
        return board_found,bx,by,mask
    
class CompetitionStateMachine:
    """
    云台调整
    PnP误差
    """
    def __init__(self, uart_sender:UART_Sender,pnpsolve:PnP_solve):
        self.pnp = pnpsolve
        self.uart = uart_sender
        self.state = 1 # 0:静止 1:中心 2:沿边巡线
        self.target_state = 0  # 0: 关闭激光 1: 开启激光
        self.return_state = 0  # 0: 停止 1: 运行 
        self.first_point = True
        self.smooth_error_x = 0.0
        self.smooth_error_y = 0.0

    def step(self, target_result, laser_found, laser_bx, laser_by, frame,current_state):
        error_x = 0.0
        error_y = 0.0
        self.state = current_state
        
        if not target_result['found']:
            if self.uart:
                self.uart.target_lost()
            cv2.putText(frame, "Target Lost: S9999,9999,0,1,0.5E", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.first_point = True
            return

        if laser_found: #有激光点
            row_x = laser_bx
            row_y = laser_by

            if self.first_point:
                self.smooth_error_x = row_x
                self.smooth_error_y = row_y
                self.first_point = False
            else:
                self.smooth_error_x = 0.7 * self.smooth_error_x + 0.3 * row_x
                self.smooth_error_y = 0.7 * self.smooth_error_y + 0.3 * row_y

            error_x = self.smooth_error_x
            error_y = self.smooth_error_y
        else: #没有激光点 pnp预测
            predict_bx, predict_by = self.pnp.laser_hit(target_result['rvec'], target_result['tvec'])
            if (predict_bx is not None and predict_by is not None and 
                not np.isnan(predict_bx) and not np.isnan(predict_by) and 
                not np.isinf(predict_bx) and not np.isinf(predict_by)):
                if predict_bx is not None and predict_by is not None:
                    error_x = predict_bx
                    error_y = predict_by

                    point_pnp = np.array([[predict_bx, predict_by, 0.0]], dtype=np.float32)
                    img_pts, _ = cv2.projectPoints(point_pnp, target_result['rvec'], target_result['tvec'],\
                                                self.pnp.camera_matrix, self.pnp.dist)
                    pred_px, pred_py = img_pts[0][0][0], img_pts[0][0][1]
                    
                    if not np.isnan(pred_px) and not np.isnan(pred_py) and not np.isinf(pred_px) and not np.isinf(pred_py):
                        cv2.drawMarker(frame, (int(pred_px), int(pred_py)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                        cv2.putText(frame, f"Laser_World: ({error_x:.1f}, {error_y:.1f})mm", \
                                (int(pred_px)+15, int(pred_py)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            else:
                error_x = target_result['offset_x']
                error_y = target_result['offset_y']

            
            self.first_point = True

        # self.state 直接取前面算的error 即为回中偏差
        if self.state == 1:
            if abs(error_x) < 10.0 and abs(error_y) < 10.0:
                    self.target_state = 1
                    self.state = 0
            else:
                    self.target_state = 0

            if abs(error_x) < 2.0: error_x = 0.0
            if abs(error_y) < 2.0: error_y = 0.0

        # 发送偏差
        self.uart.send_error(error_x,error_y,self.target_state,self.state,target_result['dist_z']) #发送偏差数据
        cv2.putText(frame, f"TX: S{error_x:.1f},{error_y:.1f},{self.target_state},{self.state},{target_result['dist_z']:.1f}E", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    model_path = "model/model_293643.mud" 
    if not os.path.exists(model_path):
        print("模型文件不存在:", os.path.abspath(model_path))
        print("请确认 model/model_293643.mud 已经和 main.py 一起上传到板端当前目录。")
        return

    detector = nn.YOLOv5(model=model_path, dual_buff=True) #并发处理 返回的是上一帧
    
    cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(),fps=60)
    dis = display.Display()

    uart = UART_Sender(baudrate=115200)
    
    pnp_engine = PnP_solve(rect_width=215.0, rect_length=305.0, focal_length=600.0)
    target_detector = TargetDetector(pnp_engine)
    laser_detector = LaserDetector(pnp_engine)
    brain = CompetitionStateMachine(uart,pnp_engine)
    fps = 0.0
    pending_frame = None
    
    try:
        while not app.need_exit():
            start_time = cv2.getTickCount()
            img = cam.read()
            object_state = detector.detect(img, conf_th=0.5, iou_th=0.45)
            current_frame = image.image2cv(img, ensure_bgr=True, copy=True)# 转成 OpenCV 的 BGR 矩阵

            # 第一轮还没有上一帧图像
            if pending_frame is None:
                pending_frame = current_frame
                continue

            # object_state 对应 pending_frame
            frame = pending_frame
            pending_frame = current_frame

            former_state = uart.receive()
            if former_state is not None:
                current_state = former_state #运动状态
            else:
                current_state = brain.state
            
            target_result = target_detector.detect(frame,object_state) #pnp返回值
            laser_found , laser_bx , laser_by ,hsv = laser_detector.detect(frame, frame, target_result) #具体运动参数

            if target_result['found']:
                center_line = target_result['center_line']

                if center_line is not None:
                    cv2.drawContours(
                        frame,
                        [center_line],
                        -1,
                        (0, 255, 0),
                        3
                    )

                # if (
                #     target_result['rvec'] is not None
                #     and target_result['tvec'] is not None
                # ):
                #     pnp_engine.draw_axes(
                #         frame,
                #         target_result['rvec'],
                #         target_result['tvec'],
                #         axes_length=100.0
                # )

            brain.step(target_result, laser_found, laser_bx, laser_by,frame,current_state) #传输参数
            
            current_time = cv2.getTickCount()
            instant_fps = cv2.getTickFrequency() / (current_time - start_time)
            if fps:
                fps = 0.90 * fps + 0.10 * instant_fps
            else:
                fps = instant_fps
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out_img = image.cv2image(frame, bgr=True, copy=True)
            dis.show(out_img)
    finally:
        uart.close()

if __name__ == '__main__':
    main()

