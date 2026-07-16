from ultralytics import YOLO
import torch

def main():
    print(f"基础模型 yolov8n.pt")

    # 1. 载入官方模型
    model = YOLO('yolov8n.pt')

    
    results = model.train(
        data=r'D:\Git\23年E题\dataset\data.yaml',  
        epochs=100,                                
        imgsz=480,                                 
        batch=16,                                  
        workers=0,                                 
        device=device,                             
        name='board_detect_result'                 
    )
    
    print("\n模型训练完成")

if __name__ == '__main__':
    main()
