import serial
import time
import threading
import queue

class UART_Sender:
    def __init__(self,port='COM3',baudrate=115200):
        """
        异步多线程初始化
        串口初始化
        默认为COM3,波特率115200
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None

        self.send_queue = queue.Queue(Maxsize=5)

        self.new_receive = None #接受并存储运动状态
        self.lock = threading.lock() #互斥锁

        self.is_running = True
        self.connect()

        self.worker_thread = threading.Thread(target=self.uart_worker_loop, daemon=True)
        self.worker_thread.start()

    def connect(self):
        """
        连接串口
        """
        try:
            self.ser = serial.Serial(self.port,self.baudrate,timeout=0.01,write_timeout=0.01)
        except serial.SerialException as e:
            print(f"connect error:{e}")
            self.ser = None

    def uart_worker_loop(self):
        """
        子线程独立运行  不断发送和
        """
        while self.is_running:
            if not (self.ser and self.ser.is_open):
                time.sleep(0.5)
                self.connect()
                continue

            #读取 发送数据
            try:
                send_data = self.send_queue.get_nowait()
                if send_data:
                    self.ser.write(send_data('utf-8'))
            except queue.Empty:
                pass
            except serial.SerialException as e:
                print(f"send error:{e}")
            
            # 接受 存储数据
            try:
                if self.ser.in_waiting > 0:
                    raw_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if len(raw_line) > 0:
                        with self.lock:
                            self.new_receive = int(raw_line)
            except Exception:
                pass

            time.sleep(0.001)

    def send_error(self,error_x,error_y,target_state,return_state):
        """
        异步发包
        主循环调用 error存到self.send_queue
        """

        #测试数据
        # print(f"error :({error_x},{error_y},{target_state},{return_state})")
        
        data = f"S{error_x},{error_y},{target_state},{return_state}E\n"
        data_bytes = data.encode('utf-8')

        try:
            if self.send_queue.full:
                self.send_queue.get_nowait() #丢弃过时数据 防止写入大于读取

            self.send_queue.put_nowait(data_bytes) #存入新数据

        except queue.Full:
            pass

    def target_lost(self):
        """
        目标丢失时发送9999,9999
        """
        self.send_error(9999,9999,0,1)

    def close(self):
        """
        关闭串口
        """
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.2)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def receive(self):
        """"
        异步收包
        主循环调用receive
        """
        with self.lock:
            state = self.new_receive()
            self.new_receive = None
            return state
        

"""独立监测"""
if __name__ == "__main__":
    print("test serial")
    uart = UART_Sender(port='COM3',baudrate=115200)

    uart.send_error(-30,20,0,1)
    time.sleep(1)

    uart.send_error(9999,9999,0,0)

    uart.close()
    print("test end")