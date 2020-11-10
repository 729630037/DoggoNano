import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append('/home/nano/minitaur-nano')
#print(sys.path)
from math import pi as PI, degrees, radians, sin, cos,sqrt,pow,atan2,acos
import math
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from envs.drivers.driver import Drive



xdata,tdata,ydata,thetadata,gammadata=[],[],[],[],[]
GaitParams={'stance_height':0.18,'down_amp':0.18,'up_amp':0.18,'flight_percent':0.5,'step_length':0.0,'freq':1.0}
USE_REINFORCEMENT_LEARNING=True
t_init=time.time()

class PositionControl:
    def __init__(self):
        self.stanceHeight=0.17
        self.downAMP=0.04
        self.upAMP=0.06
        self.flightPercent=0.5
        self.stepLength=0.15
        self.fre=2
        self.L1 = 0.09
        self.L2 = 0.162
        self._init_pose = [
            0, 0, 0, 0, 2.1, 2.1,
            2.1, 2.1
        ]


        # self.odrv0=Drive('206539A54D4D')  #1   207339A54D4D
        # self.odrv1=Drive('207339A54D4D')   #0 206539A54D4D
        # self.odrv2=Drive('206039A54D4D')  #2 206039A54D4D
        # self.odrv3=Drive('206D39A54D4D')  #3 206D39A54D4D
        self.LegGain={'kp_theta':50,'kd_theta':0.5,'kp_gamma':50,'kd_gamma':0.5}

    def Gen_signal(self,t, phase):
        gp=(t*self.fre+phase)%1
        if gp<= self.flightPercent:
            extension = -0.8 * math.sin(2 * math.pi * gp)
        else:
            extension = -0.2 * math.sin(2 * math.pi *gp)
        swing = -0.4 * math.sin(2 * math.pi * gp-PI/2) 
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
        xdata.append(x)
        ydata.append(y)      
        theta,gamma=self.CartesianToThetaGamma(x,y,leg_direction) 
        # if(odrive==self.odrv0):
        thetadata.append(theta)
        gammadata.append(gamma)
        #print(theta,gamma)
        #odrive.SetCouplePosition(theta,gamma)

    def Gait(self,t,leg0_offset=0,leg1_offset=0.5,leg2_offset=0,leg3_offset=0.5):
        if (not self.IsValidGaitParams()) or (not self.IsValidLegGain()):
            return      
        leg_direction=-1
        self.CoupledMoveLeg(t,'self.odrv0',leg0_offset,leg_direction) 
        #self.CoupledMoveLeg(t,'self.odrv1',leg1_offset,leg_direction)
        leg_direction=1    
        #self.CoupledMoveLeg(t,'self.odrv2',leg2_offset,leg_direction)
        #self.CoupledMoveLeg(t,'self.odrv3',leg3_offset,leg_direction)

    def RL_Gait(self,action):
        flag=True
        theta_gamma=self.TransformActionToThetagamma(action)
        flag=self.IsValidThetaGamma(theta_gamma)
        if not flag:
            return
        for i in range(4):
            self.odrv0.SetCouplePosition(theta_gamma[i],theta_gamma[4+i])

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
        bad=self.LegGain['kp_theta']<0 or self.LegGain['kd_theta']<0 or self.LegGain['kp_gamma']<0 or self.LegGain['kd_gamma']<0
        if bad:
            print("Invalid gains: <0.")
            return False

        bad=bad or self.LegGain['kp_theta']>320 or self.LegGain['kd_theta']>10 or self.LegGain['kp_gamma']>320 or self.LegGain['kd_gamma']>10

        if bad:
            print("Invalid gains: too high.")
            return False

        bad=bad or (self.LegGain['kp_theta']>200 and self.LegGain['kd_theta']<0.1)
        bad=bad or (self.LegGain['kp_gamma']>200 and self.LegGain['kd_gamma']<0.1)
        
        if bad:
            print("Invalid gains: underdamped.")
            return False    

        return True

    def TransformActionToThetagamma(self,action):
        '''
        minitaur:   0  2             stanford_doggo:    0  3 
                    1  3                                1  2
        action=[swing0,swing1,swing2,swing3, extension0,extension1,extension2,extension3]
        thetagamma=[theta0,theta1,theta2,theta3,gamma0,gamma1,gamma2,gamma3]
        '''
        thetagamma=[0]*8
        thetagamma[0]=action[0]
        thetagamma[1]=action[1]
        thetagamma[2]=action[3]
        thetagamma[3]=action[2]
        thetagamma[4]=PI-action[4]
        thetagamma[5]=PI-action[5]
        thetagamma[6]=PI-action[7]
        thetagamma[7]=PI-action[6]
        #print(thetagamma[0])
        return thetagamma

    def SimToReal(self,action):
        thetagamma=self.TransformActionToThetagamma(action)
        if self.IsValidThetaGamma(thetagamma):
            exit(0)
        #print(thetagamma[0],thetagamma[4])
        self.odrv0.SetCouplePosition(thetagamma[0],thetagamma[4])
        self.odrv1.SetCouplePosition(thetagamma[1],thetagamma[5])        
        self.odrv2.SetCouplePosition(thetagamma[2],thetagamma[6])
        self.odrv3.SetCouplePosition(thetagamma[3],thetagamma[7])

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

    def MotorAngleToThetaGamma(self,pos1,pos2,leg_direction):
            theta = -leg_direction*(pos1/2.0 - pos2/2.0)
            gamma = PI-leg_direction*(pos1/2.0 + pos2/2.0)
            return theta,gamma

    def ThetaGammaToMotorAngle(self,theta,gamma,leg_direction):
        if leg_direction==-1: 
            angle1=leg_direction*(PI-theta-gamma)
            angle0=leg_direction*(PI+theta-gamma)
        elif leg_direction==1:
            angle0=leg_direction*(PI-theta-gamma)
            angle1=leg_direction*(PI+theta-gamma)        
        return angle0,angle1

    def ThetaGammaToCartesian(self,theta,gamma):
        L=self.L1*cos(gamma)+sqrt(pow(self.L1*cos(gamma),2)-(pow(self.L1,2)-pow(self.L2,2)))
        x=sin(theta)*L
        y=cos(theta)*L
        return x,y

    def LToThetaGamma(self,L):
        cos_param=(pow(self.L1,2.0) + pow(L,2.0) - pow(self.L2,2.0)) / (2.0*self.L1*L)
        gamma=acos(cos_param)
        return gamma

    def IsValidThetaGamma(self,theta_gamma):
        for i in range(4):
            if theta_gamma[i]>0.8 or theta_gamma[i]<-0.8 or theta_gamma[i+4]>2.2:
                return True
        return False

    def GetThetagamma(self):
        return self.odrv0.GetThetaGamma()

if __name__=='__main__':
    x=PositionControl()
    #x.Stop()
    #time.sleep(3)
    t_init=time.time()
    st=t_init
    # while True:
    #     t=time.time()-t_init        
    #     x.Gait(t)
    #     gammadata.append(x.GetThetagamma())
    #     # tdata.append(t)
    #     # print(time.time()-st)        
    #     # st=time.time()             
    #     # time.sleep(0.01)

    for i in range(100):
        t=time.time()-t_init 
        action=x.Signal(t)
        thetagamma=x.TransformActionToThetagamma(action) 
        x1,y1=x.ThetaGammaToCartesian(thetagamma[1],thetagamma[5])      
        ydata.append(y1)
        xdata.append(x1)
        thetadata.append(thetagamma[2])
        gammadata.append(thetagamma[5])        
        tdata.append(t)
        time.sleep(0.01)

    # for i in range(50):
    #     t=time.time()-t_init 
    #     x.Gait(t)
    #     tdata.append(t)        
    #     time.sleep(0.01)



    print(time.time()-t_init )
    plt.figure()
    #plt.plot(xdata, ydata)
    #plt.plot(tdata, thetadata) 
    plt.plot(tdata, gammadata)        
    plt.show()     