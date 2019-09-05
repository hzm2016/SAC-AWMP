import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def add_final_reward(self, final_reward, steps, delay = 0):
		len_buffer = len(self.storage)
		for i in range(len_buffer - steps - delay, len_buffer - delay):
			item = list(self.storage[i])
			item[3] += final_reward
			self.storage[i] = tuple(item)

	def add_specific_reward(self, reward, idx_vec):
		for idx in idx_vec:
			idx = int(idx)
			item = list(self.storage[idx])
			item[3] += reward
			self.storage[idx] = tuple(item)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def read_table(file_name = '../../data/joint_angle.xls'):
	dfs = pd.read_excel(file_name, sheet_name='joint_angle')
	data = dfs.values[1:-1, -6:].astype(np.float)
	return data


def calc_cos_similarity(joint_angle_resample, human_joint_angle):
	joint_num = human_joint_angle.shape[1]
	dist = np.zeros(joint_num)
	for c in range(joint_num):
		dist[c] = 1 - distance.cosine(joint_angle_resample[:,c], human_joint_angle[:,c])
	return np.mean(dist)


def plot_joint_angle(joint_angle_resample, human_joint_angle):
	fig, axs = plt.subplots(human_joint_angle.shape[1])
	for c in range(len(axs)):
		axs[c].plot(joint_angle_resample[:,c])
		axs[c].plot(human_joint_angle[:,c])
	plt.legend(['walker 2d','human'])
	plt.show()


def fifo_list(data_list, value):
	data_list[:-1] = data_list[1:]
	data_list[-1] = value
	return data_list
