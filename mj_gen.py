#!/usr/bin/env python3

###############
#Author: Yi Herng Ong
#Purpose: import kinova jaco j2s7s300 into Mujoco environment
#
#("/home/graspinglab/NearContactStudy/MDP/jaco/jaco.xml")
#
###############


import gym
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import cv2
import csv
from random import random
import time
import sys
#table range(x:117~584, y:0~480)


class Kinova_MJ(object):
	def __init__(self):
		# self._model = load_model_from_path("/home/graspinglab/NearContactStudy/MDP/kinova_description/j2s7s300.xml")
		self._model = load_model_from_path("/home/leelcz/Graduate_project/mujoco_kinova_rendering/j2s7s300/j2s7s300.xml")
		
		self._sim = MjSim(self._model)
		self._viewer = MjViewer(self._sim)
		self._viewer._run_speed = 0.001
		self._timestep = 0.0001
		self._sim.model.opt.timestep = self._timestep


		self._torque = [0,0,0,0,0,0,0,0,0,0]
		self._velocity = [0,0,0,0,0,0,0,0,0,0]

		self._jointAngle = [0,0,0,0,0,0,0,0,0,0]
		self._positions = [] # ??
		self._numSteps = 0
		self._simulator = "Mujoco"
		self._experiment = "" # ??
		self._currentIteration = 0

		# define pid controllers for all joints
		#self.pid = [PID_(0.1, 0.040, 0), PID_(1.9,0.06,0), PID_(1.1,0.060,0.0),PID_(0.1,0.040,0.0), PID_(0.1,0.040,0.0), PID_(0.1,0.040,0.0),PID_(0.1,0.040,0.0), PID_(0.1,0.0,0.0), PID_(0.1,0.0,0.0), PID_(0.1,0.0,0.0)]


	#def set_step(self, seconds):
	#	self._numSteps = seconds / self._timestep
		# print(self._numSteps)

	# might want to do this function on other file to provide command on ros moveit as well
	#def set_target_thetas(self, thetas):
	#	self.pid[1].set_target_jointAngle(thetas[1])
	#	self.pid[2].set_target_jointAngle(thetas[2])
	#	self.pid[3].set_target_jointAngle(thetas[3])
	#	self.pid[4].set_target_jointAngle(thetas[4])
	#	self.pid[5].set_target_jointAngle(thetas[5])
	#	self.pid[6].set_target_jointAngle(thetas[6])
	#	self.pid[6].set_target_jointAngle(thetas[7])
	#	self.pid[6].set_target_jointAngle(thetas[8])

		# print("joint1",self.pid[1]._targetjA)

	def set_init(self):
		for i in range(40):
			self.run_mujoco(fl=1)


	def run_mujoco(self,thetas = [2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0],fl=1):
		self._sim.data.qpos[0:10] = thetas[:] 		#first 10 - first 7 are joint angles, next 3 are finger pose
		self._sim.forward()
		self._viewer.render()
#		img = self._sim.render(width=480,height=640,camera_name="camera")
		
		#img = self._sim.render(width=640,height=480,camera_name="camera2")

		#a = str(fl)
		
		#plt.imsave("testdata/img_"+a+".png",img)
		
		return 0


def degreetorad(degree):
	rad = degree/(180/math.pi)
	return rad


def read_ang_data(filename):
	f = open(filename, "r")
	if f.mode == 'r':
		angle_data = f.read()

	data = [[float(i) for i in line.split(',')] for line in angle_data.split()]
	#print(data)
    #data = [float(angle_data.split()[i]) for i in range(1,len(angle_data.split()),2)]
    #data = angle_data.split()
    #print(len(data[1]))
	return data


def img_dia(target,ori_mask,kernel,kernel2 = None):
    if type(kernel2)== type(kernel):
        kernel2 = kernel
    img_dir = target + "/img.png"
    img_r_dir = target + "/real.jpg"
    plt.imsave(img_dir, ori_mask)
    img_dilation = cv2.dilate(ori_mask, kernel, iterations=3)
    plt.imsave(target +"/img_dila.png", img_dilation)
    img_real = cv2.imread(img_r_dir)
    img_r_dilation = cv2.dilate(img_real, kernel, iterations=3)
    plt.imsave(target+"/real_img_dila.png", img_r_dilation)
    return img_dilation,img_r_dilation,img_real

if __name__ == '__main__':
	sim = Kinova_MJ()
	filename = "angle_data.csv"
	#print("working on")
	#print(filename)
	bb = read_ang_data(filename)
	#sq3193 num 658/654

	FING1 = 70
	FING2 = 72
	FING3 = 73


	for i in range(-40,len(bb)):
		if i < 0:
			ang = [2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0]
		else:
			ang = bb[i].copy()
			ang.append(FING1)
			ang.append(FING2)
			ang.append(FING3)
			ang = np.array(ang)/(180/math.pi)
			#ang[0] = 1*ang[0]
			#ang[1] = -1*ang[1]+math.pi
			#ang[2] = -1*ang[2]
			#ang[3] = 1*ang[3]
			#ang[4] = 1*ang[4]
			#ang[5] = 1*ang[5]
			#ang[6] = 1*ang[6]

			ang[0] = 1*ang[0]
			ang[1] = math.pi-ang[1]#-1+pi
			ang[2] = 1*ang[2]
			ang[3] = 1*ang[3]
			ang[4] = ang[4]
			ang[5] = ang[5]
			ang[6] = ang[6]

		print(ang)
		sim.run_mujoco(ang, fl=1)

	#bb.reshape(-1)
	#print(bb)
	#bb[1:-3] *= -1
	#ori_mask = sim.run_mujoco(bb,fl = 1)
	#cv2.imshow('My Image', ori_mask)
	#cv2.waitKey(0)






