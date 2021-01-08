import os,sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math
from envs.kinematics import Kinematics
roll_d=[]
pitch_d=[]
t1=[]
with open("dd.txt",mode='r') as f:
    for line in f:
        x, y= line.split(' ')
        roll_d.append(float(x))
        t1.append(float(y))
with open("dd1.txt",mode='r') as f1:
    for line in f1:
        x, y= line.split(' ')
        pitch_d.append(float(x))
plt.figure()
plt.plot(t1, roll_d) 
plt.plot(t1, pitch_d)     
plt.show()
k=Kinematics()
print(k._solve_IK([0.075,15,0.17],-1))

# print(a+np.random.normal(scale=0.05, size=len(a)))

