import os,sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from math import pi as PI, degrees, radians, sin, cos,sqrt,pow,atan2,acos
import math
import numpy as np

import time
from drivers.driver import Drive
import threading
import signal
import queue

from envs.gait_planner import GaitPlanner 
from envs import kinematics


xdata,tdata,ydata,thetadata,gammadata=[],[],[],[],[]
TrotGaitParams={'stance_height':0.17,'down_amp':0.04,'up_amp':0.06,'flight_percent':0.35,'step_length':0.15,'freq':2.0}
WalkGaitParams={'stance_height':0.15,'down_amp':0.04,'up_amp':0.04,'flight_percent':0.25,'step_length':0.1,'freq':2.0}
HopGaitParams={'stance_height':0.15,'down_amp':0.05,'up_amp':0.05,'flight_percent':0.2,'step_length':0.0,'freq':1.0}


flag=True

class PositionControl:
    def __init__(self,
                signal_type="ol",
                mode="trot",
                stay_still=False,
                step_frequency=2.0,
                init_theta=0.0,
                theta_amplitude=0.4,   #0.35rad=20.05度 0.3rad=17.19度
                init_gamma=1.1,
                gamma_amplitude=0.8,
                use_imu=False,
                step_length=1.5,
                step_rotation=None,
                step_angle=None,
                step_period=0.5,
                ):
        self.mode=mode
        self.theta_offset = np.zeros(4)
        self.gamma_offset = np.zeros(4)
        self.gait_planner = GaitPlanner(mode)
        self.kinematics = kinematics.Kinematics()
        self.signal_type=signal_type    
        self.stay_still = stay_still
        self.use_imu=use_imu               
        self.stanceHeight=0.17
        self.downAMP=0.04
        self.upAMP=0.06
        self.flightPercent=0.35
        self.stepLength=0.15
        self.step_frequency=step_frequency
        self.L1 = 0.09
        self.L2 = 0.162
        self._init_pose = [
            init_theta, init_theta, init_theta, init_theta, init_gamma, init_gamma,
            init_gamma, init_gamma
        ]
        self.theta_amplitude = theta_amplitude
        self.gamma_amplitude = gamma_amplitude
        self.step_length = step_length
        self.step_rotation = step_rotation
        self.step_angle = step_angle
        self.step_period = step_period
        self.odrv0_thread = None               
        self.odrv1_thread = None             
        self.odrv2_thread = None
        self.odrv3_thread = None
        self.odrives_thread= None
        self.odrive_queue=queue.Queue(maxsize=1)        
        self.ready=[0]*4
        self.LegGain=[80,0.5,50,0.5]
        self._reset_time=time.time()       

    def Signal(self, t, action):
        if self.signal_type == 'ik':
            return self.IK_signal(t, action)
        elif self.signal_type == 'ol':
            return self.OpenLoopSignal(t,action)
        else:
            return self.Gait(t)

    @staticmethod
    def _evaluate_gait_stage_coeff(current_t, action, end_t=0.0):
        # ramp function(斜坡函数)
        p = 0.8
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0

    def IK_signal(self, t, action):
        gait_stage_coeff = self._evaluate_gait_stage_coeff(t, action)
        position = np.array([0,0,0])
        orientation = np.array([0,0,0])
        step_length = self.step_length * gait_stage_coeff
        step_rotation = (self.step_rotation if self.step_rotation is not None else 0.0)
        step_angle = self.step_angle if self.step_angle is not None else 0.0
        step_period = self.step_period

        direction = -1.0 if step_length < 0 else 1.0
        frames = self.gait_planner.loop(t,step_length, step_angle, step_rotation, step_period, direction)
        fr_angles, fl_angles, br_angles, bl_angles, _ = self.kinematics.solve(orientation, position, frames)
        signal = np.array([fl_angles[0],bl_angles[0],fr_angles[0],br_angles[0],
                    fl_angles[1],bl_angles[1],fr_angles[1],br_angles[1]])
        signal +=action     
        return signal

    def OpenLoopSignal(self, t,action):
        if self.mode=="trot":
            # Generates the leg trajectories for the two digonal pair of legs.
            gamma_first, theta_first = self.GenSignal(t, 0)
            gamma_second, theta_second = self.GenSignal(t, 0.5)

            trotting_signal = np.array([
                theta_first, theta_second, theta_second, theta_first, gamma_first,
                gamma_second, gamma_second, gamma_first
            ]) 
            signal = np.array(self._init_pose) + trotting_signal
        else:
            signal=np.array(self._init_pose)
        signal +=action
        return signal

    def GenSignal(self, t, phase):
        the_amp = self.theta_amplitude
        gam_amp = self.gamma_amplitude
        start_coeff = self._evaluate_gait_stage_coeff(t, [0.0])
        the_amp *= start_coeff
        gam_amp *= start_coeff

        gp=(t*self.step_frequency+phase)%1
        if gp<= self.flightPercent:
            gamma = gam_amp * math.sin(math.pi/self.flightPercent* gp)
            theta = the_amp* math.cos(math.pi/self.flightPercent* gp) 
        else:
            percentBack = (gp-self.flightPercent)/(1.0-self.flightPercent)
            gamma = (-1+gam_amp)* math.sin(math.pi*percentBack)
            theta = the_amp * math.cos(math.pi*percentBack+math.pi)
        return gamma, theta
       
    def TransformActionToMotorCommand(self, t, action):
        if self.stay_still:
            return self._init_pose
        action = np.array(self.Signal(t,action))
        action=[action[0],action[1],-action[3],-action[2],action[4],action[5],action[7],action[6]]
        self.IsValidLegLength(action)
        return action

    def SetParams(self,GaitParams=TrotGaitParams):
        self.stanceHeight=GaitParams['stance_height']
        self.downAMP=GaitParams['down_amp']
        self.upAMP=GaitParams['up_amp']
        self.flightPercent=GaitParams['flight_percent']
        self.stepLength=GaitParams['step_length']
        self.step_frequency=GaitParams['freq']

    def SetGain(self,kp_theta,kd_theta,kp_gamma,kd_gamma):
        self.LegGain=[kp_theta,kd_theta,kp_gamma,kd_gamma]

    def Sintrajectory(self,t,gait_offset):
        gp=(t*self.step_frequency+gait_offset)%1
        if gp<= self.flightPercent:
            x=gp/self.flightPercent*self.stepLength-self.stepLength/2
            y = -self.upAMP*sin(PI*gp/self.flightPercent) + self.stanceHeight
        else:
            percentBack = (gp-self.flightPercent)/(1.0-self.flightPercent)
            x = -percentBack*self.stepLength + self.stepLength/2.0
            y = self.downAMP*sin(PI*percentBack) + self.stanceHeight
        return x,y

    def CoupledMoveLeg(self,t,gait_offset,leg_direction):
        x,y=self.Sintrajectory(t,gait_offset)
        # xdata.append(x)
        # ydata.append(y)
        L = pow((pow(x,2.0) + pow(y,2.0)), 0.5)
        cos_param = (pow(self.L1,2.0) + pow(L,2.0) - pow(self.L2,2.0)) / (2.0*self.L1*L)
        theta = atan2(leg_direction * x, y)
        if(cos_param>=1 or cos_param<=-1):
            raise ValueError("cos_param is out of bounds.")
        gamma = np.arccos(cos_param)        
        return theta,gamma

    def Gait(self,t,leg0_offset=0,leg1_offset=0.5,leg2_offset=0,leg3_offset=0.5):
        leg_direction=-1
        theta0,gamma0=self.CoupledMoveLeg(t,leg0_offset,leg_direction) 
        theta1,gamma1=self.CoupledMoveLeg(t,leg1_offset,leg_direction)
        leg_direction=1    
        theta2,gamma2=self.CoupledMoveLeg(t,leg2_offset,leg_direction)
        theta3,gamma3=self.CoupledMoveLeg(t,leg3_offset,leg_direction)
        thetagamma=[theta0,theta1,-theta3,-theta2,gamma0,gamma1,gamma3,gamma2]
        return thetagamma

    def IsValidLegLength(self,action):
        maxL=0.24
        minL=0.08
        for i in range(4):            
            x,y=self.kinematics.solve_K([action[i],action[i+4]])
            L = pow((pow(x,2.0) + pow(y,2.0)), 0.5)
            # print(L)
            if L>maxL or L<minL:
                print("The length of leg is out of bound!!!")
                exit(0)        

    def ODrive0Init(self):        
        self.odrv0=Drive('206539A54D4D')  #1   207339A54D4D
        self.odrv0.SetCoupleGain(self.LegGain)
        self.odrv0.SetCouplePosition(0,1.4) 
        self.ready[0]=1 

    def ODrive1Init(self):       
        self.odrv1=Drive('207339A54D4D')   #0 206539A54D4D        
        self.odrv1.SetCoupleGain(self.LegGain)
        self.odrv1.SetCouplePosition(0,1.4)
        self.ready[1]=1                    

    def ODrive2Init(self):                
        self.odrv2=Drive('206039A54D4D')  #2 206039A54D4D        
        self.odrv2.SetCoupleGain(self.LegGain)
        self.odrv2.SetCouplePosition(0,1.4)
        self.ready[2]=1                    

    def ODrive3Init(self):               
        self.odrv3=Drive('206D39A54D4D')  #3 206D39A54D4D
        self.odrv3.SetCoupleGain(self.LegGain)
        self.odrv3.SetCouplePosition(0,1.4)
        self.ready[3]=1                            

    def ODriveRun(self):
        while True:
            theta_gamma=self.odrive_queue.get()
            self.odrv0.SetCouplePosition(theta_gamma[0],theta_gamma[4])       
            self.odrv1.SetCouplePosition(theta_gamma[1],theta_gamma[5])        
            self.odrv2.SetCouplePosition(theta_gamma[2],theta_gamma[6])        
            self.odrv3.SetCouplePosition(theta_gamma[3],theta_gamma[7])        
            # self.odrive_queue.task_done()

    def is_valid_thetagamma(self,theta_gamma):
        for i in range(4):
            if theta_gamma[i]>0.8 or theta_gamma[i]<-0.8 or theta_gamma[i+4]>2.2:
                return True
        return False

    def Start(self):        
        self.odrv0_thread = threading.Thread(target=self.ODrive0Init,daemon=True)
        self.odrv1_thread = threading.Thread(target=self.ODrive1Init,daemon=True)
        self.odrv2_thread = threading.Thread(target=self.ODrive2Init,daemon=True)
        self.odrv3_thread = threading.Thread(target=self.ODrive3Init,daemon=True)
        self.odrives_thread = threading.Thread(target=self.ODriveRun,daemon=True)        
        self.odrv0_thread.start()
        self.odrv1_thread.start()
        self.odrv2_thread.start()
        self.odrv3_thread.start()

    def StopThread(self):
        self.odrv0_thread.join()
        self.odrv1_thread.join()
        self.odrv2_thread.join()
        self.odrv3_thread.join()       

    def Run(self,t,action):
        theta_gamma=self.TransformActionToMotorCommand(t,action)
        if self.is_valid_thetagamma(theta_gamma):
            exit(0)        
        self.odrv0.SetCouplePosition(theta_gamma[0],theta_gamma[4])       
        self.odrv1.SetCouplePosition(theta_gamma[1],theta_gamma[5])        
        self.odrv2.SetCouplePosition(theta_gamma[2],theta_gamma[6])        
        self.odrv3.SetCouplePosition(theta_gamma[3],theta_gamma[7])

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
    pos_control=PositionControl()
    pos_control.Start()
    # fd=open("123.txt",mode='w',encoding='utf-8')      
    while pos_control.ready!=[1]*4 :
        pass
    t=input("please input t:")
    while t!='t':
        pass
    time.sleep(3)
    t_init=time.time()
    st=t_init
    #pos_control.SetParams(WalkGaitParams)
    while flag:
        t=time.time()-t_init
        pos_control.Gait(t)    #walk    
        # action=pos_control.Signal(t)
        # pos_control.Run(action)
        #print(time.time()-st)
        # observation=imu.ReadDataMsg()
        #theta,gamma = pos_control.GetThetaGamma()
        # for i in range(4):
        #     fd.write(str(observation[i])+' ')        
        # # fd.write(str(theta)+' '+str(gamma)+' ')
        # fd.write(str(round(t,2))+'\n')                        
        # st=time.time()        
        # #time.sleep(0.02)
        # if t>8:
        #     break           
    pos_control.Stop()

