from maix import err, pinmap, sys, uart

class UART_Sender:
    def __init__(self, port=None, baudrate=115200):
        """
        串口初始化
        MaixCAM2 默认使用 A21/A22 对应的 /dev/ttyS4。
        MaixCAM / MaixCAM-Pro 默认使用 A19/A18 对应的 /dev/ttyS1。
        """
        device_id = sys.device_id()
        if port is None:
            port = "/dev/ttyS4" if device_id == "maixcam2" else "/dev/ttyS1"

        pin_functions = {
            "/dev/ttyS1": {"A19": "UART1_TX", "A18": "UART1_RX"},
            "/dev/ttyS4": {"A21": "UART4_TX", "A22": "UART4_RX"},
        }
        for pin, func in pin_functions.get(port, {}).items():
            err.check_raise(
                pinmap.set_pin_function(pin, func),
                f"Failed set pin {pin} function to {func}",
            )

        print("UART device:", port, "baudrate:", baudrate)
        self.serial = uart.UART(port, baudrate)
        self.target_lost_time = 0


    def send_error(self,error_x,error_y,target_state,return_state,dist_z):
        data = f"S{error_x:.1f},{error_y:.1f},{target_state},{return_state},{dist_z:.1f}E\n"

        if self.serial:
            self.serial.write_str(data)

    def target_lost(self):
        """
        目标丢失时发送9999,9999
        """
        self.send_error(9999,9999,0,1,0.5)

    def close(self):
        """
        关闭串口
        """
        if self.serial:
             self.serial.close()

    def receive(self):
        if self.serial:
            try:
                data_bytes = self.serial.read()
                if data_bytes:
                    data = data_bytes.decode('utf-8', errors='ignore').strip()
                    if data:
                        return int(data)
            except Exception as e:
                print("UART receive error:", e)
        return None
