import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
roll_d=[]
pitch_d=[]
t1=[]
# with open("dd.txt",mode='r') as f:
#     for line in f:
#         x, y= line.split(' ')
#         roll_d.append(float(y))
#         t1.append(float(x))
# plt.figure()
# plt.plot(t1, roll_d) 
# # plt.plot(t1, theta1)     
# plt.show()

a=[0.01,0.02,0.03,0.04]
a[0:1]=a[1:0]
print(a)
# print(a+np.random.normal(scale=0.05, size=len(a)))

