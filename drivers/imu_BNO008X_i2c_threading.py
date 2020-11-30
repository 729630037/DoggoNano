# SPDX-FileCopyrightText: 2020 Bryan Siepert, written for Adafruit Industries
#
# SPDX-License-Identifier: Unlicense
import time
import board
import busio
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
import math
import threading



class IMU:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA, frequency=200000)
        self.bno = BNO08X_I2C(i2c,address=0x4b)
        self.bno.enable_feature(adafruit_bno08x.BNO_REPORT_LINEAR_ACCELERATION)
        self.bno.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)
        self.bno.enable_feature(adafruit_bno08x.BNO_REPORT_GYROSCOPE)

        self.lock=threading.Lock()
        self.thread=None
        self.condition=threading.Condition()
        self.queue=[] 
        self.max_size=4   

        self.start_time = time.time()
        self.timeout=0.1
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
        if norm==0:
            return 0,0,0
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

    def size(self):
        self.lock.acquire()
        size=len(self.queue)
        self.lock.release()
        return size


    def push(self):
        if self.max_size !=0 and self.size()>self.max_size:
            raise("thread error!")
        self.lock.acquire()

        # self.dt=round(time.time()-self.then,3)  
        # self.then=time.time()  
        gyro_x, gyro_y, _ = self.bno.gyro         #陀螺仪
        # linear_accel_x,linear_accel_y,linear_accel_z= self.bno.linear_acceleration         #线性加速度 
        quat_i, quat_j, quat_k, quat_real = self.bno.quaternion                            #四元数
        
        Q=self.QuaternionMultiply([quat_real,quat_i, quat_j, quat_k])
        roll,pitch,_=self.QuaternionToEuler(Q)

        pitch=round(pitch ,2)
        roll=round(roll,2)
        roll_vel=gyro_x
        pitch_vel=gyro_y
        self.queue.append([pitch,roll,pitch_vel,roll_vel])
        self.lock.release()
        self.condition.acquire()
        self.condition.notify()
        self.condition.release()

def pop(self,block=False,timeout=0):
    if self.size() ==0:
        if block:
            self.condition.acquire()
            self.condition.wait(timeout=timeout)
            self.condition.release()
        else:
            return None
    if self.size() ==0:
        return None    
    self.lock.acquire()
    item=self.queue.pop()
    self.lock.release()
    return item

    




if __name__=='__main__':
    imu=IMU()
    while True:
        data=imu.DataHandle()
        print("Roll:%f, Pitch:%f, RollVel:%f, PitchVel:%f, XVel:%f."%(data[0],data[1],data[2],data[3],imu.x_vel))
        time.sleep(0.02)