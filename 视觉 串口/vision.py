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
"""

import cv2
import numpy as np
from uart_driver import UART_Sender
from pnp_solve import PnP_solve

uart = UART_Sender(port='COM18', baudrate=115200) # 初始化串口   记得修改端口

"""
云台运动状态 默认为0滞空
"""
current_state = 0 #下位机传入的运动状态
return_state = 0 #停止
movement_modes = 0
target_state = 0 # 为0关闭激光 为1开启激光

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
    寻找内外方框，计算中心坐标与多边形角点
    解算PnP姿态
    """
    def __init__(self,pnpsolve:PnP_solve):
        self.pnp = pnpsolve
        self.last_M = None
        self.keeptime = 0
        self.kernel = np.ones((5,5),dtype=np.uint8)

    def detect(self,frame):
        length , width = frame.shape[:2]

        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # 转化为灰度图
        gray_Biur_img = cv2.GaussianBlur(gray_img,(5,5),0)

        #二值化
        """
        A 采用大津法
        """
        _ , binary = cv2.threshold(gray_Biur_img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        """
        B 采用局部自适应    光线不均匀的时候采用
        """
        # binary = cv2.adaptiveThreshold(gray_Biur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize=11,C=2)

        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel,iterations=2) #闭运算去除微小噪点
        contours, _ = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #找出轮廓

        valid_rects = [] #收集最后内外框

        for contour in contours:
            area = cv2.contourArea(contour) #计算每个轮廓的面积
            if area > 1000: #过滤小的噪点
                perimeter = cv2.arcLength(contour,True) #计算周长 True强行闭合
                approx = cv2.approxPolyDP(contour,0.02*perimeter,True) #多边形拟合

                if len(approx) == 4: #如果拟合的结果恰好为4，即刚好为矩形线框
                    valid_rects.append((area, approx))
        
        #按面积排 大到小
        valid_rects.sort(key=lambda x:x[0],reverse=True)

        result = {
            'found': False,
            'center_x': int(width / 2), 'center_y': int(length / 2),
            'dist_z': 0.0,'offset_x':0.0,'offset_y':0.0,
            'yaw': 0.0, 'pitch': 0.0,'roll':0.0,
            'rvec': None, 'tvec': None,
            'binary':binary,
        }

        if len(valid_rects) >= 2:
            outer_approx = valid_rects[0][1] #外框
            inner_approx = valid_rects[1][1] #内框

            #内外框顶点排序
            ordered_outer = order_points(outer_approx)
            ordered_inner = order_points(inner_approx)

            #中线
            center_line = (ordered_outer + ordered_inner) / 2.0
            center_line = center_line.astype("int32")#中线顶点坐标  此时为浮点数

            center_x = int(np.mean(center_line[:,0])) #中线中心点x坐标
            center_y = int(np.mean(center_line[:,1])) #中线中心点y坐标

            # cv2.drawContours(frame,[outer_approx],-1,(255,0,0),1) #外框
            # cv2.drawContours(frame,[inner_approx],-1,(0,0,255),1) #内框
            cv2.drawContours(frame,[center_line],-1,(0,255,0),3) #中线
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)#中心点

            #pnp姿态解算
            success, dist_z, offset_x, offset_y, yaw, pitch, roll, rvec, tvec = self.pnp.solve(ordered_outer)
            if success:
                self.pnp.draw_axes(frame,rvec,tvec,axes_length=100.0)

            self.keep_time = cv2.getTickCount()

            result.update({
            'found': True,
            'center_x': center_x, 'center_y': center_y,
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

    def detect(self, frame, result):
        blurred = cv2.GaussianBlur(frame,(5,5),0) #高斯模糊去噪
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV) # 转化为HSV
        
        # 蓝紫色激光
        lower = np.array([100, 50, 50], dtype=np.uint8)
        upper = np.array([160, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 开运算去细白颗粒
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) # 闭运算填补可能的空缺

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found = False
        px, py = 0, 0
        bx, by = 0.0, 0.0
        
        for contour in contours:
            if cv2.contourArea(contour) > 20: # 确认发光斑
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    found = True
                    cv2.drawMarker(frame, (px, py), (0, 255, 0), cv2.MARKER_STAR, 18, 2)
                    break
                    
        # 换算至pnp
        if found and result['found'] and result['rvec'] is not None:
            bx , by = self.pnp.point_board(px, py, result['rvec'], result['tvec'])
            if bx is not None and by is not None:
                cv2.putText(frame, f"Laser_World: ({bx:.1f}, {by:.1f})mm", \
                            (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
        return found,bx,by
    
class CompetitionStateMachine:
    """
    云台调整
    PnP误差
    """
    def __init__(self, uart_sender:UART_Sender,pnpsolve:PnP_solve):
        self.pnp = pnpsolve
        self.uart = uart_sender
        self.state = 0 # 0:静止 1:中心 2:沿边巡线
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
            cv2.putText(frame, "Target Lost: S9999,9999,0,1E", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.first_point = True
            return

        if self.state == 1:
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
                if predict_bx is not None and predict_by is not None:
                    error_x = predict_bx
                    error_y = predict_by
                else:
                    error_x = target_result['offset_x']
                    error_y = target_result['offset_y']

                self.first_point = True

        if abs(error_x) < 15.0 and abs(error_y) < 15.0:
                self.target_state = 1
                self.state = 0
        else:
                self.target_state = 0

        if abs(error_x) < 2.0: error_x = 0.0
        if abs(error_y) < 2.0: error_y = 0.0

        # 发送偏差
        self.uart.send_error(error_x,error_y,self.target_state,self.state,target_result['dist_z']) #发送偏差数据
        cv2.putText(frame, f"TX: S{error_x},{error_y},{self.target_state},{self.state},{target_result['dist_z']}E", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    uart = UART_Sender(port='COM18', baudrate=115200)
    cap = cv2.VideoCapture(0)
    
    pnp_engine = PnP_solve(rect_width_mm=215.0, rect_length_mm=305.0, focal_length=600.0)
    target_detector = TargetDetector(pnp_engine)
    laser_detector = LaserDetector(pnp_engine)
    brain           = CompetitionStateMachine(uart,pnp_engine)
    
    while True:
        start_time = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret: 
            break
        former_state = uart.receive()
        if former_state is not None:
            current_state = former_state #运动状态
        else:
            current_state = brain.state
        
        target_result = target_detector.detect(frame) #pnp返回值返回值
        laser_found , laser_bx , laser_by = laser_detector.detect(frame, target_result) #具体运动参数
        brain.step(target_result, laser_found, laser_bx, laser_by,frame,current_state) #传输参数
        
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera PnP", frame)
        cv2.imshow("Binary", target_result['binary'])
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    uart.close()

if __name__ == '__main__':
    main()

