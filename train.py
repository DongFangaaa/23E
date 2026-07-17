from ultralytics import YOLO
import torch

def main():
    print(f"基础模型 yolov5n.pt")
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == '0':
        print(f"GPU 显卡型号: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU")
    # 载入官方模型
    model = YOLO('yolov5n.pt')

    
    results = model.train(
        data=r'D:\Git\23年E题\dataset\data.yaml',  
        epochs=100,                                
        imgsz=480,                                 
        batch=16,                                  
        workers=0,                                 
        device=device,                             
        name='light_board'                 
    )
    
    print("\n模型训练完成")

if __name__ == '__main__':
    main()
