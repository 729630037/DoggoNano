import odrive
from odrive.enums import *

import os
import time
import sys, traceback
from serial.serialutil import SerialException
from serial import Serial
import numpy as np
import datetime
#import thread


class Drive:
    def __init__(self,serial_num):
        #self.mutex = thread.allocate_lock()
        self.odrv=odrive.find_any(serial_number=serial_num)
        print('Successfully connecting Odrive'+format(serial_num))
        #self.serial_number=serial_number
        self.SetCurrentLimit(50)
        self.SetCurrentLimit(50)
    
    def SetCouplePosition(self,sp_theta,sp_gamma):
        self.odrv.axis0.controller.set_coupled_setpoints(sp_theta,sp_gamma)
        self.odrv.axis1.controller.set_coupled_setpoints(sp_theta,sp_gamma)        
        pass

    def SetCoupleGain(self,leg_gain):
        kp_theta=leg_gain[0]
        kd_theta=leg_gain[1]
        kp_gamma=leg_gain[2]
        kd_gamma=leg_gain[3]
        self.odrv.axis0.controller.set_coupled_gains(kp_theta,kd_theta,kp_gamma,kd_gamma)
        self.odrv.axis1.controller.set_coupled_gains(kp_theta,kd_theta,kp_gamma,kd_gamma)
        pass

    def SetCurrentLimit(self,current_lim):
        self.odrv.axis0.motor.config.current_lim=current_lim
        self.odrv.axis1.motor.config.current_lim=current_lim
        pass

    def GetThetaGamma(self):
        theta=self.odrv.axis0.controller.theta
        gamma=self.odrv.axis0.controller.gamma
        return theta ,gamma