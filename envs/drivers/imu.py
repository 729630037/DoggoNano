import select
import mtdevice
import math
import pdb
from math import pi, radians
import numpy as np


class XSensDriver(object):

	def __init__(self):

		device = '/dev/ttyUSB0'
		baudrate = 115200
        
		print("MT node interface: %s at %d bd."%(device, baudrate))
		self.mt = mtdevice.MTDevice(device, baudrate)

		# Will configure the device on startup
		configure_on_startup = True

		# Output data rate of device. Valid values: 1, 2, 4, 5, 10, 20, 40, 50, 80, 100, 200, 400, 800
		# Refer to MT Low Level Communication Protocol  Documentation for supported ODRs for each MTi series
		# https://base.xsens.com/hc/en-us/articles/207003759-Online-links-to-manuals-from-the-MT-Software-Suite
		odr = 10

		# Output data mode
		# 1: sensor data
		# 2. sensor data w/ rate quantities
		# 3: filter estimates
		output_mode = 2

		# MTi-30/300 [General:39, High_map_dep:40, Dynamic:41, Low_mag_dep:42, VRU_general:43]
		xkf_scenario = 39

		if configure_on_startup:
			print('Setting ODR (%d) and output mode (%d)' % (odr, output_mode))
			if odr not in [1, 2, 5, 10, 20, 40, 50, 80, 100, 200, 400]:
				raise Exception('Invalid ODR configuraton requested')
			if output_mode not in [1, 2, 3]:
				raise Exception('Invalid output mode requested')
			self.mt.configureMti(odr, output_mode)
			self.mt.ChangeBaudrate(baudrate)
			self.mt.SetCurrentScenario(xkf_scenario)
			self.mt.GoToMeasurement()
		else:
			print('Using saved odr and output configuration')		

	def get_imu_date(self):
				
		# get data
		data = self.mt.read_measurement()

		# get data (None if not present)
		#temp = data.get('Temp')	# float
        #need the roll,pitch, and the angular velocities of the base along these two axes.
		orient_data = data.get('Orientation Data')
		gyr_data = data.get('Angular Velocity')

		velocity_data = data.get('Velocity')
		position_data = data.get('Latlon')
		altitude_data = data.get('Altitude')
		acc_data = data.get('Acceleration')
		mag_data = data.get('Magnetic')
		pressure_data = data.get('Pressure')
		time_data = data.get('Timestamp')
		gnss_data = data.get('Gnss PVT')
		status = data.get('Status')	# int


		imu_data=[]
															
		if gyr_data:
			if 'Delta q0' in gyr_data: # found delta-q's
				print("they are not styles we need!")
			elif 'gyrX' in gyr_data: # found rate of turn
				gyrX = gyr_data['gyrX']
				gyrY = gyr_data['gyrY']
				gyrZ = gyr_data['gyrZ']              
			else:
				raise MTException("Unsupported message in XDI_AngularVelocityGroup.")
		
		if orient_data:
			if 'Q0' in orient_data:
				w = orient_data['Q0']
				x = orient_data['Q1']
				y = orient_data['Q2']
				z = orient_data['Q3']
				print("they are not styles we need!")              
			elif 'Roll' in orient_data:
				roll = orient_data['Roll']
				pitch = orient_data['Pitch']
				yaw = orient_data['Yaw']
			else:
				raise MTException('Unsupported message in XDI_OrientationGroup')
		
			imu_data.extend([roll,pitch,gyrX,gyrY])
			imu=np.array(imu_data)

			return imu



