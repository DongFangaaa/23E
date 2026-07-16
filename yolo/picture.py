import cv2
import os

os.makedirs('data', exist_ok=True)
cap = cv2.VideoCapture(r'C:\Users\29907\Videos\test\bfa2373ac636421dec36f19eda3c0398.mp4') # 或传 0/1 用实时摄像头录拍
count = 0
num = 113

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 每隔 10 帧抽一张保存
    if count % 7== 0:
        cv2.imwrite(f'data/img_{num}.jpg', frame)
        num += 1
    count += 1

cap.release()
print("视频帧抽取完毕")