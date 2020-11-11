import serial
import time 
import numpy as np
IMU_INDEX = 2
IMU_YAW_LSB = 3
IMU_YAW_MSB = 4
IMU_PITCH_LSB = 5
IMU_PITCH_MSB = 6
IMU_ROLL_LSB = 7
IMU_ROLL_MSB = 8
IMU_ACC_X_LSB = 9
IMU_ACC_X_MSB = 10
IMU_ACC_Y_LSB = 11
IMU_ACC_Y_MSB = 12
IMU_ACC_Z_LSB = 13
IMU_ACC_Z_MSB = 14

class IMU:
    def __init__(self, port, baudrate=115200):
        self.device = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )
        self.start_time = time.time()
        self.length=19
        self.timeout=0.1
        self.imu_data={'index':0,'yaw':0,'pitch':0,'roll ':0,'acc_x':0,'acc_y':0,'acc_z':0}
        self.device.reset_input_buffer()
        self.dt=0.0
        self.then=time.time()
        self.pitch_pre=None
        self.roll_pre=None
        self.pitch_latest=0.0
        self.roll_latest=0.0
        self.pitch_vel=0.0
        self.roll_vel=0.0
        self.x_vel=0.0
        self.y_vel=0.0
        self.x_vel_pre=0.0
        self.y_vel_pre=0.0                
        self.x_distance=0.0
        self.y_distance=0.0

        # x_dis, x_vel, x_acc, y_dis, y_vel, y_acc, z_dis, z_vel, z_acc, roll, roll_vel, pitch, pitch_vel, yaw, yaw_vel
        self.doggo_imu=[0]*15       


    def HeadCheck(self,buf):
        return (buf[0]==0xAA and buf[1]==0xAA)         
        
    def SumCheck(self,buf):
        check = 0
        for i in range(2,self.length-1):
            check +=buf[i]
        return ((check & 0xFF) == buf[self.length-1])

    def TwosComplement(self,hexstr,bits):
        value = hexstr &0xFF
        if value & (1 << (bits-1)):
            value -= 1 << bits
        return value
    
    def TwoBytesToWord(self,one,two):
        value = (one<<8 |two) & 0xFFFF
        if value & (1 << 15):
            value -= 1 << 16
        return value

    def DataHandle(self,buf):
        self.dt=round(time.time()-self.then,3)

        YAW =   0.01  * self.TwoBytesToWord(buf[IMU_YAW_MSB],buf[IMU_YAW_LSB])
        PITCH = 0.01 * self.TwoBytesToWord(buf[IMU_PITCH_MSB],buf[IMU_PITCH_LSB])
        ROLL = 0.01 * self.TwoBytesToWord(buf[IMU_ROLL_MSB],buf[IMU_ROLL_LSB])
        XACCEL = 0.001 * self.TwoBytesToWord(buf[IMU_ACC_X_MSB],buf[IMU_ACC_X_LSB])
        YACCEL = 0.001 * self.TwoBytesToWord(buf[IMU_ACC_Y_MSB],buf[IMU_ACC_Y_LSB])
        ZACCEL = 0.001 * self.TwoBytesToWord(buf[IMU_ACC_Z_MSB],buf[IMU_ACC_Z_LSB])

        # yawMultiplier = 2.506963788300836 if YAW < 0 else 2.542372881355932
        # pitchMultiplier = 2.486187845303867 if PITCH < 0 else 2.54957507082153
        # rollMultiplier = 2.510460251046025 if ROLL < 0 else 2.542372881355932

        self.imu_data['yaw'] = YAW 
        self.imu_data['pitch'] = PITCH 
        self.imu_data['roll'] =  ROLL 
        self.imu_data['acc_x'] =   XACCEL
        self.imu_data['acc_y'] =   YACCEL
        self.imu_data['acc_z'] =   ZACCEL

        self.pitch_latest=round(PITCH ,2)
        self.roll_latest=round(ROLL,2)

        if(self.pitch_pre == self.pitch_latest or self.pitch_pre==None):
            self.pitch_vel=0.0
        else:
            self.pitch_vel=round((self.pitch_latest-self.pitch_pre)/self.dt,2)

        if(self.roll_pre == self.roll_latest  or self.roll_pre==None):
            self.roll_vel=0.0
        else:
            self.roll_vel=round((self.roll_latest-self.roll_pre)/self.dt,2)

        self.x_vel=self.x_vel_pre+round(XACCEL-0.02,2)*self.dt
        self.y_vel=self.y_vel_pre+round(YACCEL+0.15,2)*self.dt
        self.x_distance += self.x_vel*self.dt
        self.y_distance += self.y_vel*self.dt        

        self.x_vel_pre=self.x_vel
        self.y_vel_pre=self.y_vel
        self.pitch_pre=self.pitch_latest
        self.roll_pre=self.roll_latest

        #self.doggo_imu[]=

        self.then=time.time()

        return [-self.pitch_latest, self.roll_latest, self.x_vel, self.y_vel]

    def read_data_msg(self,buf=bytearray()):
        """Low-level MTData receiving function.
        Take advantage of known message length."""
        #print(self.device.inWaiting())
        while True:
            for c in self.device.read():                  
                buf.append(c)
                if(len(buf)==self.length):
                    #print(buf)
                    if(self.HeadCheck(buf) and self.SumCheck(buf)):
                        imu_data_use=self.DataHandle(buf)
                        buf.clear()
                        self.device.flushInput()
                        #print(self.device.inWaiting())
                        return imu_data_use

                                  
if __name__=='__main__':
    imu=IMU("/dev/ttyTHS1")
    t1=time.time()
    for i in range(100000):
        print(imu.read_data_msg())
        time.sleep(0.01)
    imu.device.close()