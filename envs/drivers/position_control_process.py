import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from  math import pi as PI
import math 
import time
from envs.drivers.driver import Drive
from multiprocessing import Process, Queue, current_process

xdata,tdata,ydata,thetadata,gammadata=[],[],[],[],[]
GaitParams={'stance_height':0.18,'down_amp':0.18,'up_amp':0.18,'flight_percent':0.5,'step_length':0.0,'freq':1.0}
USE_REINFORCEMENT_LEARNING=True

_init_pose = [
      0, 0, 0, 0, 2, 2,
      2, 2
]
def _gen_signal(t, phase):
  period = 1 / 2   #0.5(trot)
  extension = 0.15 * math.cos(2 * math.pi / period * t + phase) # 0.35*cos(4*pi*t+phase) (trot)
  swing = 0.4 * math.sin(2 * math.pi / period * t + phase)  # 0.3*sin(4*pi*t+phase) (trot)
  return extension, swing

def _signal(t):
  # Generates the leg trajectories for the two digonal pair of legs.
  ext_first_pair, sw_first_pair = _gen_signal(t, math.pi/4)
  ext_second_pair, sw_second_pair = _gen_signal(t, math.pi+math.pi/4)

  trotting_signal = [
      sw_first_pair, sw_second_pair, sw_second_pair, sw_first_pair, ext_first_pair,
      ext_second_pair, ext_second_pair, ext_first_pair
  ]
  signal = [_init_pose[i]+trotting_signal[i] for i in range(0,len(trotting_signal))]
  return signal

class PositionControl:
    def __init__(self):
        self.ready=[0]*4            
        self.odrv0=Drive('206539A54D4D')  #1   207339A54D4D
        self.odrv1=Drive('207339A54D4D')   #0 206539A54D4D
        self.odrv2=Drive('206039A54D4D')  #2 206039A54D4D
        self.odrv3=Drive('206D39A54D4D')  #3 206D39A54D4D

        self.alive = False                      # 当 alive 为 True，线程会进行
        self.wait_end = None                    # 用来控制主线程
        self.process_odrv0 = None               
        self.process_odrv1 = None             
        self.process_odrv2 = None
        self.process_odrv3 = None

        self.odrv0_queue=Queue()
        self.odrv1_queue=Queue()        
        self.odrv2_queue=Queue()
        self.odrv3_queue=Queue()

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

    def sim_to_real0(self):
        t=time.time()     
        while True:
            print(time.time()-t)
            t=time.time()            
            theta_gamma0=self.odrv0_queue.get()
            self.odrv0.SetCouplePosition(theta_gamma0[0],theta_gamma0[1])

            #self.theta.append(-self.odrv0.GetThetaGamma())

    def sim_to_real1(self):                 
        while True:
            theta_gamma1=self.odrv1_queue.get()
            self.odrv1.SetCouplePosition(theta_gamma1[0],theta_gamma1[1])
            #self.theta1.append(-self.odrv1.GetThetaGamma())

    def sim_to_real2(self):                   
        while True:
            theta_gamma2=self.odrv2_queue.get()
            self.odrv2.SetCouplePosition(theta_gamma2[0],theta_gamma2[1])
            #self.theta2.append(-self.odrv2.GetThetaGamma())

    def sim_to_real3(self):                   
        while True:
            theta_gamma3=self.odrv3_queue.get()
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

    def start(self):
        self.process_odrv0 = Process(target=self.sim_to_real0)
        self.process_odrv1 = Process(target=self.sim_to_real1)
        self.process_odrv2 = Process(target=self.sim_to_real2)
        self.process_odrv3 = Process(target=self.sim_to_real3)

        self.process_odrv0.start()
        self.process_odrv1.start()
        self.process_odrv2.start()
        self.process_odrv3.start()

    def wait(self):
        if not self.wait_end is None:
            self.wait_end.wait()            # 阻塞主线程


    def run(self,action):
        theta_gamma=self.transform_action_to_thetagamma(action)
        self.odrv0_queue.put([theta_gamma[0],theta_gamma[4]])
        self.odrv1_queue.put([theta_gamma[1],theta_gamma[5]])
        self.odrv2_queue.put([theta_gamma[2],theta_gamma[6]])
        self.odrv3_queue.put([theta_gamma[3],theta_gamma[7]])

if __name__=='__main__':
    pos_contorl=PositionControl()
    pos_contorl.Stop()
    pos_contorl.start()
    t_init=time.time()
    st=t_init
    while True:
        t=time.time()-t_init        
        action=_signal(t)
        #print(t)
        pos_contorl.run(action)
        time.sleep(0.01)           
