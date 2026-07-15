import cv2
import numpy as np
import json

class PnP_solve:
    def __init__(self,rect_width=215.0,rect_length=305.0,focal_length=600.0,center_point=(320.0,240.0)):
        """
        初始化 3D 世界物理坐标系与相机参数
        rect_width_mm:  矩形短边真实物理宽度
        rect_length_mm: 矩形长边真实物理长度
        focal_length:   相机主光轴等效像素焦距
        center_point:   图像传感器像素中心
        """
        self.width = float(rect_width)
        self.length = float(rect_length)
        half_w = self.width/2
        half_l = self.length/2

        # 设置保底兜底参数（防止 JSON 不存在时报错）
        self.laser_origin = np.array([[35.0], [20.0], [0.0]], dtype=np.float64)
        self.laser_direction = np.array([[0.0],  [0.0],  [1.0]], dtype=np.float64)
        # 读取实测标定数据 JSON
        self.load_laser_calib("laser_calib.json")

        self.object_points = np.array([
            [-half_l,-half_w,0],
            [half_l ,-half_w,0],
            [half_l ,half_w ,0],
            [-half_l,half_w ,0],
        ],dtype=np.float32)#顺时针  从左上开始 以中心点为0，0

        #相机内部参数矩阵
        self.camera_matrix = np.array([
            [focal_length,0           ,center_point[0]],
            [0           ,focal_length,center_point[1]],
            [0           ,0           ,1              ],
        ],dtype=np.float64)

        #畸变系数向量
        self.dist = np.zeros((5,1),dtype=np.float64)

    def load_laser_calib(self, json_path="laser_calib.json"):
        """
        加载由getparm.py测得的参数
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.laser_origin = np.array(data["origin"], dtype=np.float64).reshape(3, 1)
            self.laser_direction = np.array(data["direction"], dtype=np.float64).reshape(3, 1)
            print("成功加载参数")
        except Exception as e:
            print("参数加载失败")

    def solve(self,image_point):
        """
        核心解算方法：将画面中的 4 个顶点映射到真实空间
        
        image_points: numpy 矩阵，尺寸为 (4, 2)，顺时针
        return: (success, dist_z, offset_x, offset_y, yaw_deg, pitch_deg, roll_deg, rvec, tvec)    毫米
                 - success: 是否成功解算 (bool)
                 - dist_z: 相机距离靶面正前方的直达距离
                 - offset_x/offset_y: 相机中心偏离靶心水平/垂直偏位
                 - yaw_deg: 水平偏航度数 (正为偏右，负为偏左)
                 - pitch_deg: 垂直俯仰度数 (正为下俯，负为上仰)
                 - roll_deg: 镜头横滚斜歪度数
        """
        pts_img = image_point.reshape((4,2)).astype(np.float32)

        success,rvec,tvec = cv2.solvePnP(self.object_points,pts_img,self.camera_matrix,\
                                         self.dist,flags=cv2.SOLVEPNP_IPPE)
        
        if not success:
            return False,0.0,0.0,0.0,0.0,0.0,0.0,None,None
        
        #平移向量提取(tvec :3*1)
        #tvec[0] 为真实 X 位移，tvec[1] 为真实 Y 位移，tvec[2] 为深度测距 Z
        offset_x = float(tvec[0][0])
        offset_y = float(tvec[1][0])
        dist_z = float(tvec[2][0])

        #旋转向量提取
        rot_mat , _ = cv2.Rodrigues(rvec)

        # 利用旋转矩阵元素反解标准航向度数 (Yaw 偏航角,Pitch 俯仰角,Roll 横滚角)   弧度
        yaw_rad = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        pitch_rad = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2))	
        roll_rad  = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        
        #角度
        yaw_deg = float(np.degrees(yaw_rad))
        pitch_deg = float(np.degrees(pitch_rad))
        roll_deg = float(np.degrees(roll_rad))

        return True,dist_z,offset_x,offset_y,yaw_deg,pitch_deg,roll_deg,rvec,tvec
    
    #将HSV得到的激光点坐标换到pnp后的坐标
    def point_board(self,point_x,point_y,rvec,tvec):
        if rvec is None or tvec is None:
            return None,None
        
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        t = tvec.reshape(3, 1)

        # 像素坐标转为齐次向量并逆乘内参，得到镜头视线方向
        pixel_vec = np.array([[float(point_x)], [float(point_y)], [1.0]], dtype=np.float64)
        d_cam = np.linalg.inv(self.camera_matrix) @ pixel_vec

        # 将视线方向与镜头原点转入PnP物理坐标系
        d_world = R_inv @ d_cam #在摄像头坐标系的视线方向换到黑板上的方向
        O_world = R_inv @ (-t) #摄像头坐标系原点在黑板坐标系的位置
        if abs(d_world[2, 0]) < 1e-10:
            return None, None # 射线平行于黑板，无交点
        
        # 计算视线撞击在黑板平面 Z=0 上的精确缩放倍率 s 与实体触点
        s = -O_world[2, 0] / d_world[2, 0]
        P_hit = O_world + s * d_world

        # 返回在黑板上的坐标
        return float(P_hit[0, 0]), float(P_hit[1, 0])

    def laser_hit(self,rvec,tvec):
        """
        关闭激光依据pnp算出的坐标直接找点
        """
        if rvec is None or tvec is None:
            return None,None
        
        #pnp得到激光应该的坐标
        R , _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        t = tvec.reshape(3,1)

        laser_origin = np.array(self.laser_origin,dtype=np.float64).reshape(3,1)
        laser_direction = np.array(self.laser_direction,dtype=np.float64).reshape(3,1)

        O_world = R_inv @ (laser_origin - t)
        d_world = R_inv @ laser_direction
        if abs(d_world[2, 0]) < 1e-10:
            return None,None
        
        s = -O_world[2, 0] / d_world[2, 0]
        P_hit = O_world + s * d_world
        return float(P_hit[0, 0]), float(P_hit[1, 0]) # 返回pnp预测落点

    def draw_axes(self,frame,rvec,tvec,axes_length=100.0):
        """
        X Y Z分别为红 绿 蓝
        """

        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(frame,self.camera_matrix,self.dist,rvec,tvec,axes_length,3)

    def update_camera_matrix(self, focal_length, center_x, center_y):
        """
        动态修改内参矩阵
        """
        self.camera_matrix[0, 0] = float(focal_length)
        self.camera_matrix[1, 1] = float(focal_length)
        self.camera_matrix[0, 2] = float(center_x)
        self.camera_matrix[1, 2] = float(center_y)