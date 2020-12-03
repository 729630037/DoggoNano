import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from math import pi as PI, degrees, radians, sin, cos,sqrt,pow,atan2,acos
import math

import time
from drivers.driver import Drive
import threading
import queue
import signal

xdata,tdata,ydata,thetadata,gammadata=[],[],[],[],[]
TrotGaitParams={'stance_height':0.17,'down_amp':0.04,'up_amp':0.06,'flight_percent':0.35,'step_length':0.15,'freq':2.0}
WalkGaitParams={'stance_height':0.15,'down_amp':0.04,'up_amp':0.04,'flight_percent':0.25,'step_length':0.1,'freq':2.0}
HopGaitParams={'stance_height':0.15,'down_amp':0.05,'up_amp':0.05,'flight_percent':0.2,'step_length':0.0,'freq':1.0}
USE_REINFORCEMENT_LEARNING=True

lock = threading.Lock()
flag=True

class PositionControl:
    def __init__(self):
        self.stanceHeight=0.17
        self.downAMP=0.04
        self.upAMP=0.06
        self.flightPercent=0.35
        self.stepLength=0.15
        self.fre=2
        self.L1 = 0.09
        self.L2 = 0.162
        self._init_pose = [
            0, 0, 0, 0, 2.1, 2.1,
            2.1, 2.1
        ]

        self.thread_odrv0 = None               
        self.thread_odrv1 = None             
        self.thread_odrv2 = None
        self.thread_odrv3 = None
        self.ready=[0]*4
        self.odrv0_queue=queue.Queue()
        self.odrv1_queue=queue.Queue()        
        self.odrv2_queue=queue.Queue()
        self.odrv3_queue=queue.Queue()
        self.LegGain=[80,0.5,50,0.5]

    def SetParams(self,GaitParams=TrotGaitParams):
        self.stanceHeight=GaitParams['stance_height']
        self.downAMP=GaitParams['down_amp']
        self.upAMP=GaitParams['up_amp']
        self.flightPercent=GaitParams['flight_percent']
        self.stepLength=GaitParams['step_length']
        self.fre=GaitParams['freq']

    def SetGain(self,kp_theta,kd_theta,kp_gamma,kd_gamma):
        self.LegGain=[kp_theta,kd_theta,kp_gamma,kd_gamma]

    def Gen_signal(self,t, phase):
        fre = 2  #2(trot)
        gp=(t*fre+phase)%1
        if gp<= self.flightPercent:
            extension = -0.8 * math.sin(math.pi/self.flightPercent* gp)
            swing = 0.4 * math.cos(math.pi/self.flightPercent* gp) 
        else:
            percentBack = (gp-self.flightPercent)/(1.0-self.flightPercent)
            extension = 0.2 * math.sin(math.pi*percentBack)
            swing = 0.4 * math.cos(math.pi*percentBack+PI) 
        return extension, swing

    def Signal(self,t):
        # Generates the leg trajectories for the two digonal pair of legs.
        ext_first_pair, sw_first_pair = self.Gen_signal(t, 0)
        ext_second_pair, sw_second_pair = self.Gen_signal(t, 0.5)

        trotting_signal = [
            sw_first_pair, sw_second_pair, sw_second_pair, sw_first_pair, ext_first_pair,
            ext_second_pair, ext_second_pair, ext_first_pair
        ]
        signal = [self._init_pose[i]+trotting_signal[i] for i in range(0,len(trotting_signal))]
        return signal


    def CartesianToThetaGamma(self,x,y,leg_direction):
        L = pow((pow(x,2.0) + pow(y,2.0)), 0.5)
        cos_param = (pow(self.L1,2.0) + pow(L,2.0) - pow(self.L2,2.0)) / (2.0*self.L1*L)
        theta = atan2(leg_direction * x, y)
        if(cos_param<=1 and cos_param>=-1):
            gamma = acos(cos_param)
        return theta,gamma

    def Sintrajectory(self,t,gait_offset):
        gp=(t*self.fre+gait_offset)%1
        if gp<= self.flightPercent:
            x=gp/self.flightPercent*self.stepLength-self.stepLength/2
            y = -self.upAMP*sin(PI*gp/self.flightPercent) + self.stanceHeight
        else:
            percentBack = (gp-self.flightPercent)/(1.0-self.flightPercent)
            x = -percentBack*self.stepLength + self.stepLength/2.0
            y = self.downAMP*sin(PI*percentBack) + self.stanceHeight
        return x,y

    def CoupledMoveLeg(self,t,odrive,gait_offset,leg_direction):
        x,y=self.Sintrajectory(t,gait_offset)
        # xdata.append(x)
        # ydata.append(y)
        theta,gamma=self.CartesianToThetaGamma(x,y,leg_direction)
        #gammadata.append(gamma)
        #print(theta,gamma)
        odrive.SetCouplePosition(theta,gamma)

    def Gait(self,t,leg0_offset=0,leg1_offset=0.5,leg2_offset=0,leg3_offset=0.5):
        if (not self.IsValidGaitParams()) or (not self.IsValidLegGain()):
            return      
        leg_direction=-1
        self.CoupledMoveLeg(t,self.odrv0,leg0_offset,leg_direction) 
        self.CoupledMoveLeg(t,self.odrv1,leg1_offset,leg_direction)
        leg_direction=1    
        self.CoupledMoveLeg(t,self.odrv2,leg2_offset,leg_direction)
        self.CoupledMoveLeg(t,self.odrv3,leg3_offset,leg_direction)

    def IsValidGaitParams(self):
        maxL=0.25
        minL=0.08
        if (self.stanceHeight+self.downAMP)>maxL or sqrt(pow(self.stanceHeight,2))+pow(self.stepLength/2,2)>maxL:
            print("Gait overextends leg")
            return False

        if self.stanceHeight-self.upAMP<minL:
            print("Gait underextends leg")
            return False

        if self.flightPercent <= 0 or self.flightPercent > 1.0:
            print("Flight percent is invalid");
            return False

        if self.fre < 0:
            print("Frequency cannot be negative")
            return False

        if self.fre > 10.0:
            print("Frequency is too high (>10)")
            return False

        return True

    def IsValidLegGain(self):
        bad=self.LegGain[0]<0 or self.LegGain[1]<0 or self.LegGain[2]<0 or self.LegGain[3]<0
        if bad:
            print("Invalid gains: <0.")
            return False

        bad=bad or self.LegGain[0]>320 or self.LegGain[1]>10 or self.LegGain[2]>320 or self.LegGain[3]>10

        if bad:
            print("Invalid gains: too high.")
            return False

        bad=bad or (self.LegGain[0]>200 and self.LegGain[1]<0.1)
        bad=bad or (self.LegGain[2]>200 and self.LegGain[3]<0.1)
        
        if bad:
            print("Invalid gains: underdamped.")
            return False    

        return True

    def TransformActionToThetagamma(self,action):
        '''
        minitaur:   0  2             stanford_doggo:    0  3 
                    1  3                                1  2
        action=[swing0,swing1,swing2,swing3, extension0,extension1,extension2,extension3]
        theta_gamma=[theta0,theta1,theta2,theta3,gamma0,gamma1,gamma2,gamma3]
        '''
        theta_gamma=[0]*8
        theta_gamma[0]=action[0]
        theta_gamma[1]=action[1]
        theta_gamma[2]=-action[3]
        theta_gamma[3]=-action[2]
        theta_gamma[4]=PI-action[4]
        theta_gamma[5]=PI-action[5]
        theta_gamma[6]=PI-action[7]
        theta_gamma[7]=PI-action[6]
        if self.is_valid_thetagamma(theta_gamma):
            exit(0)
        return theta_gamma

    def sim_to_real0(self):
        self.odrv0=Drive('206539A54D4D')  #1   207339A54D4D
        self.odrv0.SetCoupleGain(self.LegGain)
        self.odrv0.SetCouplePosition(0,1.4) 
        self.ready[0]=1 
        t=time.time()     

    def sim_to_real1(self):
        self.odrv1=Drive('207339A54D4D')   #0 206539A54D4D        
        self.odrv1.SetCoupleGain(self.LegGain)
        self.odrv1.SetCouplePosition(0,1.4)
        self.ready[1]=1                    


    def sim_to_real2(self):
        self.odrv2=Drive('206039A54D4D')  #2 206039A54D4D        
        self.odrv2.SetCoupleGain(self.LegGain)
        self.odrv2.SetCouplePosition(0,1.4)
        self.ready[2]=1                    

    def sim_to_real3(self):
        self.odrv3=Drive('206D39A54D4D')  #3 206D39A54D4D
        self.odrv3.SetCoupleGain(self.LegGain)
        self.odrv3.SetCouplePosition(0,1.4)
        self.ready[3]=1                     
         
    def is_valid_thetagamma(self,theta_gamma):
        for i in range(4):
            if theta_gamma[i]>0.8 or theta_gamma[i]<-0.8 or theta_gamma[i+4]>2.2:
                return True
        return False

    def Start(self):
        self.alive = True
        self.wait_end = threading.Event()
        
        self.thread_odrv0 = threading.Thread(target=self.sim_to_real0)
        self.thread_odrv0.setDaemon(True)                        # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv1 = threading.Thread(target=self.sim_to_real1)
        self.thread_odrv1.setDaemon(True)                       # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv2 = threading.Thread(target=self.sim_to_real2)
        self.thread_odrv2.setDaemon(True)                       # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv3 = threading.Thread(target=self.sim_to_real3)
        self.thread_odrv3.setDaemon(True)                       # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv0.start()
        self.thread_odrv1.start()
        self.thread_odrv2.start()
        self.thread_odrv3.start()

    def Run(self,action):
        theta_gamma=self.TransformActionToThetagamma(action)
        self.odrv0.SetCouplePosition(theta_gamma[0],theta_gamma[4])       
        self.odrv1.SetCouplePosition(theta_gamma[1],theta_gamma[5])        
        self.odrv2.SetCouplePosition(theta_gamma[2],theta_gamma[6])        
        self.odrv3.SetCouplePosition(theta_gamma[3],theta_gamma[7])
        # self.odrv0_queue.put([theta_gamma[0],theta_gamma[4]])
        # self.odrv1_queue.put([theta_gamma[1],theta_gamma[5]])
        # self.odrv2_queue.put([theta_gamma[2],theta_gamma[6]])
        # self.odrv3_queue.put([theta_gamma[3],theta_gamma[7]])

    def GetThetaGamma(self):
        return self.odrv0.GetThetaGamma()

    def Stop(self):
        self.odrv0.SetCoupleGain(self.LegGain)
        self.odrv1.SetCoupleGain(self.LegGain)
        self.odrv2.SetCoupleGain(self.LegGain)
        self.odrv3.SetCoupleGain(self.LegGain)
        self.odrv0.SetCouplePosition(0,1.4)
        self.odrv1.SetCouplePosition(0,1.4)
        self.odrv2.SetCouplePosition(0,1.4)
        self.odrv3.SetCouplePosition(0,1.4)
        pass

    def CommandAllLegs(self,theta,gamma):
        self.odrv0.SetCoupleGain(self.LegGain)
        self.odrv1.SetCoupleGain(self.LegGain)
        self.odrv2.SetCoupleGain(self.LegGain)
        self.odrv3.SetCoupleGain(self.LegGain)
        self.odrv0.SetCouplePosition(theta,gamma)
        self.odrv1.SetCouplePosition(theta,gamma)
        self.odrv2.SetCouplePosition(theta,gamma)
        self.odrv3.SetCouplePosition(theta,gamma) 

    def Hop(self):
        self.SetParams(GaitParams=HopGaitParams)
        theta,gamma=self.CartesianToThetaGamma(0,self.stanceHeight-self.upAMP,1)
        self.SetGain(120,2,20,2)
        self.CommandAllLegs(theta,gamma)
        time.sleep(0.2)
        theta,gamma=self.CartesianToThetaGamma(0,self.stanceHeight+self.downAMP,1)
        self.SetGain(120,1,80,1)
        self.CommandAllLegs(theta,gamma)
        time.sleep(0.2)
        theta,gamma=self.CartesianToThetaGamma(0,self.stanceHeight,1)
        self.SetGain(120,2,20,2)
        self.CommandAllLegs(theta,gamma)                       
        time.sleep(0.6)


def handler(signum, frame):
    global flag
    flag=False

if __name__=='__main__':
    signal.signal(signal.SIGINT,handler)
    pos_contorl=PositionControl()
    pos_contorl.Start()
    # imu=imu_BNO008X_uart.IMU("/dev/ttyTHS1")
    # imu.DeviceInit()
    # fd=open("123.txt",mode='w',encoding='utf-8')      
    while pos_contorl.ready!=[1]*4 :
        pass
    t='a'
    t=input("please input t:")
    while t!='t':
        pass
    # time.sleep(3)
    t_init=time.time()
    st=t_init
    #pos_contorl.SetParams(WalkGaitParams)
    while flag:
        t=time.time()-t_init
        # pos_contorl.Gait(t)    #walk    
        action=pos_contorl.Signal(t)
        pos_contorl.Run(action)
        print(time.time()-st)
        # observation=imu.ReadDataMsg()
        #theta,gamma = pos_contorl.GetThetaGamma()
        # for i in range(4):
        #     fd.write(str(observation[i])+' ')        
        # # fd.write(str(theta)+' '+str(gamma)+' ')
        # fd.write(str(round(t,2))+'\n')                        
        st=time.time()        
        # #time.sleep(0.02)
        # if t>8:
        #     break           
    pos_contorl.Stop()

