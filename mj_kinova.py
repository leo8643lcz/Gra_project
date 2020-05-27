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
from mujoco_py import MjViewer, load_model_from_path, MjSim, MjRenderContextOffscreen
from mujoco_py.generated import const
from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import cv2
from random import random
import time
import sys
import inspect
#table range(x:117~584, y:0~480)


class Kinova_MJ(object):
	def __init__(self):
		# self._model = load_model_from_path("/home/graspinglab/NearContactStudy/MDP/kinova_description/j2s7s300.xml")
		self._model = load_model_from_path("/home/leelcz/Graduate_project/mujoco_kinova_rendering/j2s7s300/j2s7s300.xml")

		self._sim = MjSim(self._model)

		self._viewer = MjRenderContextOffscreen(self._sim, 0)
		#self._viewer = MjViewer(self._sim)
		#self._viewer.cam.fixedcamid = 2
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
#		self.pid = [PID_(0.1, 0.040, 0), PID_(1.9,0.06,0), PID_(1.1,0.060,0.0),PID_(0.1,0.040,0.0), PID_(0.1,0.040,0.0), PID_(0.1,0.040,0.0),PID_(0.1,0.040,0.0), PID_(0.1,0.0,0.0), PID_(0.1,0.0,0.0), PID_(0.1,0.0,0.0)]


	def set_step(self, seconds):
		self._numSteps = seconds / self._timestep
		# print(self._numSteps)

	# might want to do this function on other file to provide command on ros moveit as well
	#def set_target_thetas(self, thetas):
	#	self.pid[0].set_target_jointAngle(thetas[0])
#		self.pid[1].set_target_jointAngle(thetas[1])
#		self.pid[2].set_target_jointAngle(thetas[2])
#		self.pid[3].set_target_jointAngle(thetas[3])
#		self.pid[4].set_target_jointAngle(thetas[4])
#		self.pid[5].set_target_jointAngle(thetas[5])
#		self.pid[6].set_target_jointAngle(thetas[6])
#		self.pid[6].set_target_jointAngle(thetas[7])
#		self.pid[6].set_target_jointAngle(thetas[8])

		# print("joint1",self.pid[1]._targetjA)


	def run_mujoco(self,thetas = [2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0],fl=1):
		self._sim.data.qpos[0:10] = thetas[:] 		#first 10 - first 7 are joint angles, next 3 are finger pose
		self._sim.forward()
#		img = self._sim.render(width=480,height=640,camera_name="camera")
		#self._viewer.cam.type = const.CAMERA_FIXED
		#self._viewer.cam.fixedcamid = 0
		#self._viewer.render(420, 380, 0)
		#img = np.asarray(self._viewer.read_pixels(420, 380, depth=False)[::-1, :, :], dtype=np.uint8)

		#img = self._viewer._read_pixels_as_in_window()
		#img = self._sim.render(width=640,height=480,camera_name="camera2")
		self._viewer.render(640, 480, 0)
		img = np.asarray(self._viewer.read_pixels(640, 480, depth=False)[::-1, :, :], dtype=np.uint8)

		#img = self._viewer._read_pixels_as_in_window()
		#a = str(fl)
		
		#plt.imsave("testdata/img_"+a+".png",img)
		
		return img


def degreetorad(degree):
	rad = degree/(180/math.pi)
	return rad


def read_ang_data(filename):
    f = open(filename, "r")
    if f.mode == 'r':
        angle_data = f.read()
    data = [float(angle_data.split()[i]) for i in range(1,len(angle_data.split()),2)]
    return data

def read_ang_data_v(filename):
	f = open(filename, "r")
	if f.mode == 'r':
		angle_data = f.read()

	data = [[float(i) for i in line.split(',')] for line in angle_data.split()]
	#print(data)
    #data = [float(angle_data.split()[i]) for i in range(1,len(angle_data.split()),2)]
    #data = angle_data.split()
    #print(len(data[1]))
	return data


def img_dia(target,itr,ori_mask,kernel,kernel2 = None):
    if type(kernel2)== type(kernel):
        kernel2 = kernel
    #img_dir = target + "/img.png"
    img_r_dir = target + "/real{}.jpg".format(itr)
    #plt.imsave(img_dir, ori_mask)
    img_dilation = cv2.dilate(ori_mask, kernel, iterations=3)
    plt.imsave(target +"/img_dila.png", img_dilation)
    img_real = cv2.imread(img_r_dir)
    img_r_dilation = cv2.dilate(img_real, kernel, iterations=3)
    plt.imsave(target+"/real_img_dila.png", img_r_dilation)
    return img_dilation,img_r_dilation,img_real

def RGB2YUV(target,mask,real):
    mask_out = cv2.cvtColor(mask, cv2.COLOR_BGR2YUV)
    real_out = cv2.cvtColor(real, cv2.COLOR_BGR2YUV)
    plt.imsave(target+"/mask_YUV.png", mask_out)
    plt.imsave(target+"/real_YUV.png", real_out)
    return mask_out, real_out

def find_center(xi,xf,yi,yf,mask,tar_value):
	output_cel = []
	for i in range(xi,xf):
		for j in range(yi,yf):
			if np.array_equal(mask[i][j], tar_value):
				output_cel.append([i,j])
	#print(output_cel)
	return(output_cel)

def pixel_cal(mask):
	n_x, n_y = mask.shape
	out = 0
	for i in range(n_x):
		for j in range(n_y):
			if mask[j][i] != 0:
				out +=1
	return out

#def contour_dis(input, target):


def Kmeanclus(target,mask,real,parK = 5):
	Z_mask = np.float32(mask.reshape((-1, 3)))
	Z_real = np.float32(real.reshape((-1, 3)))

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = parK # modifies]d
	ret_real, label_real, center_real = cv2.kmeans(Z_real, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	ret_mask, label_mask, center_mask = cv2.kmeans(Z_mask, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	center_real = np.uint8(center_real)
	#print(center_real )
	ceter_cel = find_center(0, 479, 118, 584, real, center_real[2])
	#print(center_real)

	res_real = center_real[label_real.flatten()]
	res2_real = res_real.reshape((real.shape))
    #1023
	for i,j in ceter_cel:
		res2_real[i][j] = [0,0,0]

	for i in range(0,480):
		for j in range(117,585):
			if np.array_equal(res2_real[i][j], center_real[0]):
				res2_real[i][j] = [0,256,256]

	plt.imsave(target+"/real_KKKK"+str(K)+".png", res2_real)
	label_real_mod = label_real.reshape(480,640)
	label_real_out = label_real_mod
	for i in range(480):
		for j in range(118):
			label_real_mod[i][j] = label_real_mod[10][118]

	res_real_mod = center_real[label_real_mod]
	plt.imsave(target+"/real_K"+str(K)+"_mod.png", res_real_mod)

	center_mask = np.uint8(center_mask)
	res_mask = center_mask[label_mask.flatten()]
	res2_mask = res_mask.reshape((mask.shape))
	plt.imsave(target+"/mask_K"+str(K)+".png", res2_mask)
	label_mask_out = label_mask.reshape(480, 640)
	return label_mask_out,label_real_out,res2_real

#117 584

def cluster_matching_mask_g(target, mask_l,real_l,K,in_mask,itr):
	pixel_cal_r = np.ones(K,)
	pixel_cal_m = np.ones(K,)
	for i in range(480):
		for j in range(117, 585):
			pixel_cal_m[mask_l[i][j]] += 1
			pixel_cal_r[real_l[i][j]] += 1

	or_m = np.argsort(pixel_cal_m)
	or_r = np.argsort(pixel_cal_r)

	#print(or_r)
	out_mask = np.zeros(in_mask.shape,np.uint8)
	out_label = np.zeros(real_l.shape,np.uint8)

	threshold = int(K/2)
	or_list = or_r[0:threshold]
	#print(or_list)
	for th in range(threshold):
		for i in range(480):
			for j in range(117, 585):
				if (real_l[i][j] in or_list) and out_label[i][j] == 0:
					k = random()
					if k < 1:
						out_mask[i][j] = [255,255,255]
						out_label[i][j] = 255
					else:
						out_mask[i][j] = [0,0,0]
						out_label[i][j] = 0
				elif out_label[i][j] != 0:
					continue
				else:
					out_mask[i][j] = [0, 0, 0]
					out_label[i][j] = 0

	plt.imsave(target + "/mask_out_K" + str(itr) + ".png", out_mask)
	return out_label

def grab_cuting(target,K,nwmask,sim_img,real):
	#img = cv2.imread(target+'/real.jpg')
	img = real
	mask = np.zeros(img.shape[:2], np.uint8)
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)
	#newmask = cv2.imread(target + "/mask_out_K" + str(K) + ".png", 0)
	#newmask = cv2.imread(target + "/mask_out_K4.png", 0)
	newmask = nwmask
	#print(np.allclose(newmask,newmask1))
	#floodfill
	fill_img = nwmask.copy()
	h, w = newmask.shape[:2]
	fillmask = np.zeros((h + 2, w + 2), np.uint8)
	cv2.floodFill(fill_img, fillmask, (0, 0), 255)
	fill_img_inv = cv2.bitwise_not(fill_img)
	im_out = newmask | fill_img_inv

	#cv2.imshow("Thresholded Image", newmask)
	#cv2.imshow("Floodfilled Image", fill_img)
	#cv2.imshow("Inverted Floodfilled Image", fill_img_inv)
	#cv2.imshow("Foreground", im_out)
	# wherever it is marked white (sure foreground), change mask=1
	# wherever it is marked black (sure background), change mask=0
	newmask = im_out.copy()
	#print(newmask)

	gray_mjimage = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
	_, mj_seg = cv2.threshold(gray_mjimage, 0,255, cv2.THRESH_BINARY_INV)
	mask[newmask == 0] = 0
	mask[newmask == 255] = 1
	rect = (0, 117, 480, 584)
	mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
	mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	img = img * mask[:, :, np.newaxis]
	plt.imsave(target + "/Grab" + str(K) + ".jpg", img)
	plt.imsave(target + "/Grab_mask" + str(K) + ".jpg", mask)
	plt.imsave(target + "/Grab_mjmask" + str(K) + ".jpg", mj_seg)
	return img

def refine_sagemetation(target,mask):
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#for i in range(len(contours)):
	cnt = contours[4]
	n_x,n_y = mask.shape
	contu  = np.zeros([n_y,n_x,3], dtype = int)
	img = cv2.drawContours(contu, [cnt], 0, (255, 255, 255), 1)
	plt.imsave(target + "/contour.jpg", img)


def camara_pos_improving(sim,input,target):
	a = pixel_cal(input)
	b = pixel_cal(target)
	thes_pixel = 100
	z = sim.ca_p_z
	x = sim.ca_p_x
	y = sim.ca_p_y


	if b - a < thes_pixel:
		z += 0.01
	elif b - a > thes_pixel:
		z -= 0.01



if __name__ == '__main__':
	sim = Kinova_MJ()
	target = "testdata5"
	#filename = target+"/angle_data.txt"
	filename = target + "/angle_sequence.csv"
	print("working on")
	print(target)
	#bb = read_ang_data(filename)
	#sq3193 num 658/654
	#FING1 = 70
	#FING2 = 72
	#FING3 = 73
	#bb.append(FING1)
	#bb.append(FING2)
	#bb.append(FING3)
	#bb = np.array(bb)/(180/math.pi)
	#bb.reshape(-1)

	#bb[0] = -1*bb[0]+ math.pi/2
	#bb[1] = math.pi-bb[1]
	#print(bb)

	bb = read_ang_data_v(filename)

	FING1 = 70
	FING2 = 72
	FING3 = 73

	for i in range(len(bb)):
		ang = bb[i].copy()
		ang.append(FING1)
		ang.append(FING2)
		ang.append(FING3)
		ang = np.array(ang) / (180 / math.pi)
			# ang[0] = 1*ang[0]
			# ang[1] = -1*ang[1]+math.pi
			# ang[2] = -1*ang[2]
			# ang[3] = 1*ang[3]
			# ang[4] = 1*ang[4]
			# ang[5] = 1*ang[5]
			# ang[6] = 1*ang[6]

		ang[0] = -1 * ang[0] + math.pi / 2
		ang[1] = math.pi - ang[1]  # -1+pi
		ang[2] = ang[2]
		ang[3] = ang[3]
		ang[4] = ang[4]
		ang[5] = ang[5]
		ang[6] = ang[6]
		#sim.run_mujoco(ang, fl=1)

		ori_mask = sim.run_mujoco(ang, fl=1)
	#if ori_mask is not None:
	#	cv2.imwrite("mujoco.jpg", ori_mask)
	#cv2.imshow('My Image', ori_mask)
	#cv2.waitKey(0)

		gray_img = cv2.cvtColor(ori_mask, cv2.COLOR_BGR2GRAY)
		#plt.imsave(target+"/gray_mask.png", gray_img)
		kernel = np.ones((3,3), np.uint8)
		kernel2 = np.ones((3, 3), np.uint8)
		mask_dia,real_dia,real_data = img_dia(target,i,ori_mask,kernel,kernel2)
	#
		mask_yuv,real_yuv = RGB2YUV(target,mask_dia,real_dia)
	#for i in range(5):
	#	Kmeanclus(target,mask_yuv,real_yuv,i+2)

		mask_l, real_l, mask_full = Kmeanclus(target, mask_yuv, real_yuv, 4)
		grab_mask = cluster_matching_mask_g(target,mask_l, real_l, 4, mask_full,i)
		real = np.copy(real_data)
		realimg = cv2.imread(target + '/real{}.jpg'.format(i))
		_ = grab_cuting(target, i, grab_mask,ori_mask,realimg)

		#input = cv2.imread(target+'/img_dila.png', cv2.IMREAD_GRAYSCALE)
		#target_img = cv2.imread(target + '/Grab{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
		#_, target_img = cv2.threshold(target_img, 127, 255, 0)
	#refine_sagemetation(target,target_img)
	#print(type(input))

	#cv2.imshow('My Image', input)
		#cv2.imshow('My Image2', target_img)
		#cv2.waitKey(0)






