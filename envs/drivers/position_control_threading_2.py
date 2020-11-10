import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from  math import pi as PI

import time
from envs.drivers.driver import Drive
import threading
import queue

xdata,tdata,ydata,thetadata,gammadata=[],[],[],[],[]
GaitParams={'stance_height':0.18,'down_amp':0.18,'up_amp':0.18,'flight_percent':0.5,'step_length':0.0,'freq':1.0}
USE_REINFORCEMENT_LEARNING=True


class PositionControl:
    def __init__(self):
        self.ready=[0]*4
        self.thread_odrv0 = None               
        self.thread_odrv1 = None             
        self.thread_odrv2 = None
        self.thread_odrv3 = None


    def transform_action_to_thetagamma(self,action):
        '''
        minitaur:   0  2             stanford_doggo:    0  3 
                    1  3                                1  2
        action=[swing0,swing1,swing2,swing3, extension0,extension1,extension2,extension3]
        theta_gamma=[theta0,theta1,theta2,theta3,gamma0,gamma1,gamma2,gamma3]
        '''
        theta_gamma=[0]*8
        theta_gamma[0]=-action[0]
        theta_gamma[1]=-action[1]
        theta_gamma[2]=action[3]
        theta_gamma[3]=action[2]
        theta_gamma[4]=PI-action[4]
        theta_gamma[5]=PI-action[5]
        theta_gamma[6]=PI-action[7]
        theta_gamma[7]=PI-action[6]
        if self.IsValidThetaGamma(theta_gamma):
            exit(0)
        return theta_gamma

    def sim_to_real0(self,que):
        self.odrv0=Drive('206539A54D4D')  #1   207339A54D4D
        self.odrv0.SetCoupleGain(50,0.5,50,0.5)
        self.odrv0.SetCouplePosition(0,1.4) 
        self.ready[0]=1      
        while True:
            theta_gamma0=que.get()
            self.odrv0.SetCouplePosition(theta_gamma0[0],theta_gamma0[1])
            #self.theta.append(-self.odrv0.GetThetaGamma())

    def sim_to_real1(self,que):
        self.odrv1=Drive('207339A54D4D')   #0 206539A54D4D        
        self.odrv1.SetCoupleGain(50,0.5,50,0.5)
        self.odrv1.SetCouplePosition(0,1.4)
        self.ready[1]=1                    
        while True:
            theta_gamma1=que.get()
            self.odrv1.SetCouplePosition(theta_gamma1[0],theta_gamma1[1])
            #self.theta1.append(-self.odrv1.GetThetaGamma())

    def sim_to_real2(self,que):
        self.odrv2=Drive('206039A54D4D')  #2 206039A54D4D        
        self.odrv2.SetCoupleGain(50,0.5,50,0.5)
        self.odrv2.SetCouplePosition(0,1.4)
        self.ready[2]=1                    
        while True:
            theta_gamma2=que.get()
            self.odrv2.SetCouplePosition(theta_gamma2[0],theta_gamma2[1])
            #self.theta2.append(-self.odrv2.GetThetaGamma())

    def sim_to_real3(self,que):
        self.odrv3=Drive('206D39A54D4D')  #3 206D39A54D4D
        self.odrv3.SetCoupleGain(50,0.5,50,0.5)
        self.odrv3.SetCouplePosition(0,1.4)
        self.ready[3]=1                     
        while True:
            theta_gamma3=que.get()
            self.odrv3.SetCouplePosition(theta_gamma3[0],theta_gamma3[1])
            #self.theta3.append(-self.odrv3.GetThetaGamma())


    def Stop(self):
        self.odrv0.SetCoupleGain(50,0.5,50,0.5)
        self.odrv1.SetCoupleGain(50,0.5,50,0.5)
        self.odrv2.SetCoupleGain(50,0.5,50,0.5)
        self.odrv3.SetCoupleGain(50,0.5,50,0.5)
        self.odrv0.SetCouplePosition(0,1.4)
        self.odrv1.SetCouplePosition(0,1.4)
        self.odrv2.SetCouplePosition(0,1.4)
        self.odrv3.SetCouplePosition(0,1.4)
        pass


    def IsValidThetaGamma(self,theta_gamma):
        for i in range(4):
            if theta_gamma[i]>0.8 or theta_gamma[i]<-0.8 or theta_gamma[i+4]>2.2:
                return True
        return False

    def start(self,q1,q2,q3,q4):
        self.alive = True
        self.wait_end = threading.Event()
        
        self.thread_odrv0 = threading.Thread(target=self.sim_to_real0,args=(q1,))
        self.thread_odrv0.setDaemon(True)                        # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv1 = threading.Thread(target=self.sim_to_real1,args=(q2,))
        self.thread_odrv1.setDaemon(True)                       # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv2 = threading.Thread(target=self.sim_to_real2,args=(q3,))
        self.thread_odrv2.setDaemon(True)                       # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv3 = threading.Thread(target=self.sim_to_real3,args=(q4,))
        self.thread_odrv3.setDaemon(True)                       # 当主线程结束，读线程和主线程一并退出

        self.thread_odrv0.start()
        self.thread_odrv1.start()
        self.thread_odrv2.start()
        self.thread_odrv3.start()

    def wait(self):
        if not self.wait_end is None:
            self.wait_end.wait()            # 阻塞主线程


    def run(self,action):
        theta_gamma=self.transform_action_to_thetagamma(action)
        self.odrv0_queue.put([theta_gamma[0],theta_gamma[4]])
        self.odrv1_queue.put([theta_gamma[1],theta_gamma[5]])
        self.odrv2_queue.put([theta_gamma[2],theta_gamma[6]])
        self.odrv3_queue.put([theta_gamma[3],theta_gamma[7]])