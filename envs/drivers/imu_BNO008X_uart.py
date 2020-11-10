import serial
import time 
import numpy as np
import math
import struct
import sys
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
        self.length=8
        self.timeout=0.1
        self.device.flush()
        self.dt=0.0
        self.then=time.time()
        self.pitch=0.0
        self.roll=0.0
        self.pitch_vel=0.0
        self.roll_vel=0.0                
        self.x_vel=0.0
        self.x_acc=0.0                    
        self.x_distance=0.0

    def QuaternionMultiply(self,Q0,Q1=[0,0,-1,0]):
        w0, x0, y0, z0 = Q0
        w1, x1, y1, z1 = Q1
        return [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]

    def QuaternionToEuler(self,Q):
        """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
        norm=math.sqrt(Q[0]*Q[0]+Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3])
        w=Q[0]/norm
        x=Q[1]/norm
        y=Q[2]/norm
        z=Q[3]/norm
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # roll=math.degrees(roll)

        sinp = 2 * (w * y - z * x)
        sinp = 1.0 if sinp>1 else sinp
        sinp = -1.0 if sinp<-1 else sinp
        pitch=math.asin(sinp)
        # pitch=math.degrees(pitch)        

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # yaw=math.degrees(yaw)

        return roll, pitch, yaw

    def EulerToQuaternion(self,roll, pitch, yaw):

        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)

        return [qw, qx, qy, qz]

    def DataHandle(self,buf):
        self.dt=round(time.time()-self.then,3)

        Q=self.QuaternionMultiply([buf[0],buf[1],buf[2],buf[3]])
        roll,pitch,_=self.QuaternionToEuler(Q)

        self.pitch=round(pitch ,2)
        self.roll=round(roll,2)
        self.roll_vel=buf[4]
        self.pitch_vel=buf[5]
        self.x_acc=buf[6]
        self.x_vel=buf[7]
        self.x_distance += self.x_vel*self.dt      

        self.then=time.time()

        # return [self.roll, self.pitch, self.roll_vel, self.pitch_vel]
        return [self.pitch, self.roll, self.pitch_vel, self.roll_vel]

    def DeviceInit(self):
        self.device.flushInput()
        # while self.device.read() == b'' or self.device.read() == 0x00:
        #     pass
        for i in range(10000):
            c =self.device.readline().rstrip()
            if c>=b'-0.1':
                break                


    def ReadDataMsg(self):
        """Low-level MTData receiving function.
        Take advantage of known message length."""
        #print(self.device.inWaiting())
        buf=[]
        self.device.flushInput()        
        while True:
            c =self.device.readline().rstrip()
            if c!=b'':
                try:
                    c=float(c)           
                    buf.append(c)
                    if(len(buf)==self.length):                                        
                        return self.DataHandle(buf)
                except:
                    pass

if __name__=='__main__':
    imu=IMU("/dev/ttyTHS1")
    #imu=IMU("COM9")
    t1=time.time()
    #imu.DeviceInit()
    for i in range(1000000):
        print(imu.ReadDataMsg(),imu.x_acc,imu.x_vel)
        time.sleep(0.04)
    imu.device.close()