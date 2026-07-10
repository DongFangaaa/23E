import cv2
import numpy as np
from uart_driver import UART_Sender

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

cap = cv2.VideoCapture(1) # 连接 初始化摄像头 0为电脑摄像头

#设置显示高宽
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)  # 设置宽为500
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500) # 设置高为500


cv2.namedWindow("Trackbars",cv2.WINDOW_NORMAL) # 创建一个窗口用于放置滑动条
cv2.namedWindow("Camera",cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("Gray",cv2.WINDOW_NORMAL)

cv2.createTrackbar("H Min1", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 60, 255, nothing) #H色相 S饱和度 V明度
cv2.createTrackbar("V Min", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("H Max1", "Trackbars", 10, 179, nothing)#低段红
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("H Min2","Trackbars",169,179,nothing)#高段红
cv2.createTrackbar("H Max2", "Trackbars", 179, 179, nothing)

# 线框识别调整
cv2.createTrackbar("low","Camera",30,255,nothing)
cv2.createTrackbar("high","Camera",100,255,nothing)

smooth_x = 0.0
smooth_y = 0.0
first_point = True
last_M = None
keep_time = 0

while True:
    ret, frame = cap.read()
    start_time = cv2.getTickCount()
    former_state = uart.receive()
    if former_state is not None:
        current_state = former_state

    kernel = np.ones((5,5),np.uint8) #定义一个5x5的卷积核

    if not ret:
        print("读取失败")
        break

    #先将图像转化为灰度，再高斯模糊去噪
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # 转化为灰度图
    gray_Biur_img = cv2.GaussianBlur(gray_img,(5,5),0) #高斯模糊去噪

    #先高斯模糊去噪，再转化为HSV
    blurred = cv2.GaussianBlur(frame,(5,5),0) #高斯模糊去噪
    hsv_img = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV) # 转化为HSV
    
    #灰度寻找线框
    Low = cv2.getTrackbarPos("low","Camera") #获取当前canny参数
    High = cv2.getTrackbarPos("high","Camera")

    edges = cv2.Canny(gray_Biur_img,Low,High) #提取边缘
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,iterations=2) #闭运算 先膨胀再腐蚀 使得边缘闭合
    contours, _ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #找出轮廓

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

        cv2.drawContours(frame,[outer_approx],-1,(255,0,0),1) #外框
        cv2.drawContours(frame,[inner_approx],-1,(0,0,255),1) #内框
        cv2.drawContours(frame,[center_line],-1,(0,255,0),3) #中线
    
    #HSV找出红色激光位置
    H_Min1 = cv2.getTrackbarPos("H Min1","Trackbars") 
    S_Min = cv2.getTrackbarPos("S Min","Trackbars") 
    V_Min = cv2.getTrackbarPos("V Min","Trackbars") 
    H_Max1 = cv2.getTrackbarPos("H Max1","Trackbars") 
    S_Max = cv2.getTrackbarPos("S Max","Trackbars") 
    V_Max = cv2.getTrackbarPos("V Max","Trackbars") 
    H_Min2 = cv2.getTrackbarPos("H Min2","Trackbars") 
    H_Max2 = cv2.getTrackbarPos("H Max2","Trackbars")
    
    #低段红
    lower_bound1 = np.array([H_Min1,S_Min,V_Min]) 
    upper_bound1 = np.array([H_Max1,S_Max,V_Max]) 
    #高段红
    lower_bound2 = np.array([H_Min2,S_Min,V_Min]) 
    upper_bound2 = np.array([H_Max2,S_Max,V_Max]) 

    #最终需要调节HSV的范围来创建掩膜，目的是将特特定的红色光点提取出来
    mask1 = cv2.inRange(hsv_img,lower_bound1,upper_bound1) #根据HSV的范围创建掩膜 范围内为白255，范围外为黑0
    mask2 = cv2.inRange(hsv_img,lower_bound2,upper_bound2)

    #合并mask
    mask = cv2.bitwise_or(mask1,mask2)

    #定义一个5x5的卷积核，对HSV的掩膜进行腐蚀和膨胀操作，去除噪点
    mask = cv2.erode(mask,kernel,iterations=1) #腐蚀操作，去除小的白色噪点 防止识别到不需要的红点，但同时会使得目标变小
    mask = cv2.dilate(mask,kernel,iterations=2) #膨胀操作，第一次恢复目标大小，第二次使得目标变大，增强识别效果

    #找激光轮廓 算出重心
    contours_mask , _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    find = False #标志位 是否读取激光点
    cx,cy = 0,0

    for contour in contours_mask:
        if cv2.contourArea(contour) > 60:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                find = True #成功找到

                #在重心画一个绿色的十字准星     原始坐标
                # cv2.drawMarker(frame,(cx,cy),(0,255,0),cv2.MARKER_CROSS,20,2)
                # cv2.putText(frame,f"laser({cx},{cy})",(cx+10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)# 显示坐标


    # target_find = False #标志位 是否读取到目标
    M = None
    if len(valid_rects) >= 2:
    # 将之前识别出的中线和重心 转换到一个虚拟场中
        dst_pts = np.array([
            [0,0],
            [500,0],
            [500,500],
            [0,500], #顺时针
        ],dtype="float32")
        # target_find = True

        M = cv2.getPerspectiveTransform(center_line.astype("float32"),dst_pts) #生成转换矩阵
        last_M = M
        keep_time = cv2.getTickCount()

    elif last_M is not None and (cv2.getTickCount() - keep_time) / cv2.getTickFrequency() < 0.2:
        M = last_M
        
    if M is not None:
        if find:
            point_red = np.array([[[cx,cy]]],dtype="float32") #激光点
            point_red_pts = cv2.perspectiveTransform(point_red,M) #虚拟场中对应的激光点
            row_x = float(point_red_pts[0][0][0])
            row_y = float(point_red_pts[0][0][1]) #对应坐标

            if first_point:
                smooth_x = row_x
                smooth_y = row_y
                first_point = False
            else:
                smooth_x = 0.7 * smooth_x + 0.3 * row_x
                smooth_y = 0.7 * smooth_y + 0.3 * row_y

            x_pts = int(smooth_x)
            y_pts = int(smooth_y)

            print(f"原始({cx},{cy}) -> 虚拟场坐标: ({x_pts}, {y_pts})")
            error_x = 0
            error_y = 0 #偏差

            virtual_frame = cv2.warpPerspective(frame, M, (500, 500))
            
            #在重心画一个黄色的十字准星
            cv2.drawMarker(virtual_frame,(x_pts,y_pts),(255,255,0),cv2.MARKER_CROSS,20,2)
            cv2.putText(virtual_frame,f"laser({x_pts},{y_pts})",(x_pts+10,y_pts-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)# 显示坐标

            cv2.imshow("virtual", virtual_frame)
        else:
            first_point = True
            
        #单片机得到的偏差为正或负  x为正代表需要向右  为负代表需要向左  y为正代表需要向下 为负代表需要向上
        # error_x = 250 - x_pts
        # error_y = 250 - y_pts

        error_x = 0
        error_y = 0

        if current_state == 0: #滞空
            error_x = 0
            error_y = 0
        elif current_state == 1: #运动方式1 回到中心
            return_state = 1
            error_x = 250 - center_x
            error_y = 250 - center_y # 原系下中点偏差
            if abs(center_x-250) < 15 and  abs(center_y-250) < 15:
                current_state = 0
                return_state = 0
                target_state = 1
        elif current_state == 2 and find: #运动方式2 沿线运动 
            #从（0，0）开始
            target_state = 1
            if movement_modes == 0: #回到原点
                error_x = 0 - x_pts
                error_y = 0 - y_pts
                if x_pts < 10 and y_pts < 10:
                    movement_modes = 1
            elif movement_modes == 1: #顺时针运动
                error_x = 50
                error_y = y_pts - 0
                if x_pts >= 480:
                    movement_modes = 2
            elif movement_modes == 2:
                error_y = 50
                error_x = 500 - x_pts
                if y_pts >= 480:
                    movement_modes = 3
            elif movement_modes == 3:
                error_x = -50
                error_y = 500 - y_pts
                if x_pts < 20:
                    movement_modes = 4
            elif movement_modes == 4:
                error_y =  -50
                error_x = 0 - x_pts
                if x_pts < 15 and y_pts < 15:
                    current_state = 0
                    return_state = 0
                    movement_modes = 0

        if abs(error_x) < 5:
            error_x = 0
        if abs(error_y) < 5:
            error_y = 0

        uart.send_error(error_x,error_y,target_state,return_state) #发送偏差数据
        cv2.putText(frame, f"TX: S{error_x},{error_y},{target_state},{return_state}E", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        uart.target_lost() #目标丢失
        cv2.putText(frame, "Target Lost: S9999,9999,0,1E", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    current_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Trackbars",hsv_img)
    cv2.imshow("Camera", frame)
    #cv2.imshow("Mask",mask)
    #cv2.imshow("Gray",gray_img)
    cv2.imshow("Canny",edges)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
uart.close()