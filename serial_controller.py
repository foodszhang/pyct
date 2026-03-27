import serial
import time
import threading

class SerialController:
    def __init__(
            self, port, baudrate, parity=serial.PARITY_NONE, 
            stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=400):
        self.port = port
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.bytesize = bytesize
        self.ser = serial.Serial(
            port=self.port, baudrate=self.baudrate, 
            parity=self.parity, stopbits=self.stopbits, bytesize=self.bytesize, timeout=timeout)
        if not self.ser.isOpen():
            raise Exception('Serial port not open')
        self.ser.flushInput()
        self.ser.flushOutput()
        self.lock = threading.Lock()

    def send(self, msg):
        self.lock.acquire()
        self.ser.write(msg)
        self.lock.release()

    def read(self, sep=b'\r'):
        self.lock.acquire()
        buf = self.ser.read_until(sep)
        self.lock.release()
        return buf

    def close(self):
        self.ser.close()
#38400HIV

class UltraBrightController(SerialController):
    status_code = ['待预热', '预热中', '待开启', 'X射线开启', 'X射线过载', 'X射线无法开启', '自检中']
    def __init__(self, port, baudrate, timeout=4):
        super().__init__(port, baudrate, timeout=timeout)

    def warm_up(self):
        self.send(b'WUP\r')
        buf = self.read()
        print("warmup read line:{}".format(buf))
        if buf.startswith(b'WUP'):
            return True
        else:
            return False

    def xray_on(self):
        self.send(b'XON\r')
        buf = self.read()
        print("xray on read line:{}".format(buf))
        if buf.startswith(b'XON'):
            return True
        else:
            return False

    def xray_off(self):
        self.send(b'XOF\r')
        buf = self.read()
        print("xray on read line:{}".format(buf))
        if buf.startswith(b'XOF'):
            return True
        else:
            return False
    def query_preheat(self):
        self.send(b'SPH\r')
        buf = self.read()
        print("query read line:{}".format(buf))
        if buf.startswith(b'SPH'):
            status = buf.split()
            # status[1] 状态
            # status[2] 电压
            # status[2] 电流
            return status[1]
        else:
            return False
    def query_all(self):
        self.send(b'SAR\r')
        buf = self.read()
        print("query read line:{}".format(buf))
        if buf.startswith(b'SAR'):
            status = buf.split()
            # status[1] 状态
            # status[2] 电压
            # status[2] 电流

            return self.status_code[int(status[1])], float(status[2]), float(status[3])
        else:
            return False
    def query_status(self):
        self.send(b'STS\r')
        buf = self.read()
        print("query read line:{}".format(buf))
        if buf.startswith(b'STS'):
            status = buf.split()
            # status[1] 状态
            # status[2] 电压
            # status[2] 电流

            return status[1]
        else:
            return False

    def query_setting(self):
        self.send(b'SVI\r')
        buf = self.read()
        print("query read line:{}".format(buf))
        if buf.startswith(b'SVI'):
            status = buf.split()
            # status[1] 状态
            # status[2] 电压
            # status[2] 电流

            return status[1:]
        else:
            return False

    def set_voltage(self, voltage: int):
        self.send('HIV {}\r'.format(voltage).encode())
        buf = self.read()
        print("set_voltage read line:{}".format(buf))
        if buf.startswith(b'HIV'):
            return True
        else:
            return False


    def set_focus_mode(self, mode: int):
        '''
        0: small
        1: middle
        2: large
        '''
        self.send('CFS {}\r'.format(mode).encode())
        buf = self.read()
        print("set_focus_mode read line:{}".format(buf))
        if buf.startswith(b'CFS'):
            return True
        else:
            return False




    def set_current(self, current: int):
        self.send('CUR {}\r'.format(current).encode())
        buf = self.read()
        print("set_current read line:{}".format(buf))
        if buf.startswith(b'CUR'):
            return True
        else:
            return False



class ZolixMcController(SerialController):
    # default com4
    def __init__(self, port, baudrate, timeout=400):
        super().__init__(port, baudrate, timeout=timeout)

    def query(self):
        self.send(b'SetSpeed? X\r')
        buf = self.read()
        print("query speed read line:{}".format(buf))
        self.send(b'SetInitSpeed? X\r')
        buf = self.read()
        print("query init speed read line:{}".format(buf))
        self.send(b'SetStageDriveRat? X\r')
        buf = self.read()
        print("query stage rat read line:{}".format(buf))
        self.send(b'SetStageStepsRev? X\r')
        buf = self.read()
        print("query steps rev read line:{}".format(buf))
    def set_stage_rate(self,rat):
        self.send('SetStageDriveRat X,{}\r'.format(rat).encode())
        buf = self.read()
        print("set_rat read line:{}".format(buf))
        if buf == b'OK\r':
            return True
        else:
            return False
    def open_test(self):
        self.send(b'hello\r')
        buf = self.read()
        print("open_test read line:{}",format(buf))
        if not buf:
            print("open_test read line is empty error!")
        if buf == b'OK\r':
            return True
        else:
            return False

    def set_speed(self, speed: int):
        self.send('SetSpeed X,{}\r'.format(speed).encode())
        buf = self.read()
        print("set_speed read line:{}".format(buf))
        if not buf:
            print("set_speed read line is empty error!")
        if buf == b'OK\r':
            return True
        else:
            return False
    def set_init_speed(self, speed: int):
        self.send('SetInitSpeed X,{}\r'.format(speed).encode())
        buf = self.read()
        print("set_init_speed read line:{}".format(buf))
        if not buf:
            print("set_speed read line is empty error!")
        if buf == b'OK\r':
            return True
        else:
            return False

    def motion_rotation(self, degree: float):
        # 这里XORP直接固定了一定是X轴旋转之类的， 暂时先不深究了
        self.send('GoPosition X,O,R,P,{}\r'.format(degree).encode())
        buf = self.read()
        if buf == b'READY\r':
            buf = self.read()
            print(buf)
        print('!!!!', buf)
        if not buf:
            raise Exception("motion_rotation read line is empty error!")
        if buf.endswith(b'OK\r'):
            return True
        else:
            return False
if __name__ == '__main__':
    #ZC = ZolixMcController('COM4', 19200)
    #import time
    ##最高速5000 96s
    #ZC.set_speed(10)
    #ZC.set_init_speed(10)
    ##ZC.set_stage_rate(180)
    #ZC.query()
    #last_t = time.time()
    #ZC.motion_rotation(180)
    #print("cost time:", time.time() - last_t)
    ###360 172.733
    ###720 189.732
    #传动比180
    #每转脉冲数1600
    #脉冲当量0.00125

    UB = UltraBrightController('COM3', 38400, timeout=4)
    #print(UB.xray_off())
    #while True:
    print(UB.query_preheat())
    print(UB.query_setting())
    print(UB.set_voltage(90))
    print(UB.set_current(300))
    print(UB.warm_up())
    for i in range(10000):
     print(UB.query_all())
     time.sleep(1)
    time.sleep(1)
