import serial
import time

class UART_Sender:
    def __init__(self,port='COM3',baudrate=115200):
        """
        串口初始化
        默认为COM3,波特率115200
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connect()

    def connect(self):
        """
        连接串口
        """
        try:
            self.ser = serial.Serial(self.port,self.baudrate,timeout=0.01)
        except serial.SerialException as e:
            print(f"connect error:{e}")
            self.ser = None

    def send_error(self,error_x,error_y):
        """
        发送偏差数据
        以S<X>,<Y>E\n的格式发送
        为utf-8编码
        """

        #测试数据
        print(f"error :({error_x},{error_y})")
        
        if self.ser and self.ser.is_open:
            try:
                data = f"S{error_x},{error_y}E\n"
                self.ser.write(data.encode('utf-8'))
            except serial.SerialException as e:
                print(f"send error:{e}")

    def target_lost(self):
        """
        目标丢失时发送9999,9999
        """
        self.send_error(9999,9999)

    def close(self):
        """
        关闭串口
        """
        if self.ser and self.ser.is_open:
            self.ser.close()


"""独立监测"""
if __name__ == "__main__":
    print("test serial")
    uart = UART_Sender(port='COM3',baudrate=115200)

    uart.send_error(-30,20)
    time.sleep(1)

    uart.send_error(9999,9999)

    uart.close()
    print("test end")