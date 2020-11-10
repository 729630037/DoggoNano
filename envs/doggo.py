import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import string
import minitaur_gym_env
import time

halfpi = 1.57079632679
pai=2*halfpi
twopi = 4 * halfpi
x1=[]
y1=[]
t1=[]
theta1=[]
gamma1=[]
angle01=[]
angle11=[]


class stanford_doggo:
    def __init__(self):
        self.stanceHeight=0.15
        self.downAMP=0.02
        self.upAMP=0.02
        self.flightPercent=0.35
        self.stepLength=0.2
        self.fre=2
        self.L1 = 0.09
        self.L2 = 0.162
        self.kx_flight=1
        self.kz_flight=1
        self.xz_set_vel=[0,0]
        self.environment = minitaur_gym_env.MinitaurGymEnv(
            render=True,
            motor_velocity_limit=np.inf,
            accurate_motor_model_enabled=True,
            hard_reset=False,
            on_rack=False)

    def CartesianToThetaGamma(self,x,y,leg_direction):
        L = math.pow((math.pow(x,2.0) + math.pow(y,2.0)), 0.5)
        cos_param = (math.pow(self.L1,2.0) + math.pow(L,2.0) - math.pow(self.L2,2.0)) / (2.0*self.L1*L)
        theta = math.atan2(leg_direction * x, y)
        if(cos_param<=1 and cos_param>=-1):
            gamma = math.acos(cos_param)
        return theta,gamma

    def ThetaGammaToMotorAngle(self,theta,gamma,leg_direction):
        if leg_direction==-1: 
            angle1=leg_direction*(pai-theta-gamma)
            angle0=leg_direction*(pai+theta-gamma)
        elif leg_direction==1:
            angle0=leg_direction*(pai-theta-gamma)
            angle1=leg_direction*(pai+theta-gamma)        
        return angle0,angle1

    def Sintrajectory(self,t,pre_t,gait_offset):
        gp=(t*self.fre+gait_offset)%1
        if gp<= self.flightPercent:
            x=gp/self.flightPercent*self.stepLength-self.stepLength/2
            y = -self.upAMP*math.sin(math.pi*gp/self.flightPercent) + self.stanceHeight
        else:
            percentBack = (gp-self.flightPercent)/(1.0-self.flightPercent)
            x = -percentBack*self.stepLength + self.stepLength/2.0
            y = self.downAMP*math.sin(math.pi*percentBack) + self.stanceHeight
        return x,y

    def CoupledMoveLeg(self,t,pre_t,gait_offset,leg_direction,i,action):
        x,y=self.Sintrajectory(t,pre_t,gait_offset)
        x1.append(x)
        y1.append(y)
        theta,gamma=self.CartesianToThetaGamma(x,y,leg_direction)
        if i==1:
            theta1.append(theta)
            gamma1.append(gamma)
            #print("theta{}: {:.2f}  gamma{}: {:.2f}".format(i,theta,i,pai-gamma))
        angle0,angle1=self.ThetaGammaToMotorAngle(theta,gamma,leg_direction)
        action.append(angle0)
        action.append(angle1)

    def gait(self,t,pre_t,leg0_offset,leg1_offset,leg2_offset,leg3_offset):
        action=[]
        leg_direction=-1
        self.CoupledMoveLeg(t,pre_t,leg0_offset,leg_direction,0,action)  
        self.CoupledMoveLeg(t,pre_t,leg1_offset,leg_direction,1,action)
        leg_direction=1    
        self.CoupledMoveLeg(t,pre_t,leg2_offset,leg_direction,2,action)
        self.CoupledMoveLeg(t,pre_t,leg3_offset,leg_direction,3,action)
        #print(action)
        #action=[halfpi]*8
        obser, reward, done, _ = self.environment.step(action)
        return done

    def CalTarEndPosition(self,xz_set_vel,xz_vel):
        x_end=(self.flightPercen*xz_set_vel[0])/2.0-self.kx_flight*(xz_vel[0]-xz_set_vel[0])
        z_end=self.stanceHeight
        return x_end,z_end
    
    def cycloidTrajectory(self,t,pre_t,gait_offset,x,y):
        gp=(t*self.fre+gait_offset)%1
        observation=self.environment._get_observation()
        xz_vel=observation[28:31:2]    
        x_end,z_end=self.CalTarEndPosition(self.xz_set_vel,xz_vel)
        phi=(2*math.pi*gp)/self.flightPercent
        x=(x_end-x)*(phi-math.sin(phi))/(2*math.pi)+x
        y=z_end*(1-math.cos(phi))/2+y



if __name__=="__main__":
    doggo=stanford_doggo()
    # jump
    t = 0.0
    t_end = t + 2
    i = 0
    ref_time = time.time()
    pre_t=0
    fixedTimeStep=0.01
    while t<t_end:
        t1.append(t)
        done=doggo.gait(t,pre_t,0,0.5,0.5,0)
        if done:
            #break
            pass
        #print(obser[28:31])
        pre_t=t
        t = t + fixedTimeStep
    plt.figure()
    #print(x1)
    plt.plot(t1, gamma1)
    plt.show()    