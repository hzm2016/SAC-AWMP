# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:29:32 2019

@author: kuangen
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import cv2
import sys
sys.path.insert(0,'../')
from code.utils.utils import *
from scipy import stats, signal


def plot_error_line(t, acc_mean_mat, acc_std_mat = None, legend_vec = None,
                    marker_vec=['o', '+', 'v', 'x', 'd', '*', ''],
                    line_vec=['-', '--', '-.', ':', '-', '--', '-.'],
                    line_width_vec=[2, 2, 2, 2, 2, 2, 2], marker_size=5
                    ):
    if acc_std_mat is None:
        acc_std_mat = 0 * acc_mean_mat
    # acc_mean_mat, acc_std_mat: rows: methods, cols: time
    color_vec = plt.cm.Dark2(np.arange(8))
    for r in range(acc_mean_mat.shape[0]):
        plt.plot(t, acc_mean_mat[r, :], linestyle=line_vec[r],
                 marker=marker_vec[r], markersize=marker_size, linewidth=line_width_vec[r],
                 color=color_vec[r])
        plt.fill_between(t, acc_mean_mat[r, :] - acc_std_mat[r, :],
                         acc_mean_mat[r, :] + acc_std_mat[r, :], alpha=0.1, color=color_vec[r])
    if legend_vec is not None:
        plt.legend(legend_vec, loc = 'lower right')

def plot_test_acc():
    method_name_vec = ['', 'human_angle_still_steps', 'human_angle_still_steps_ATD3']
    # method_name_vec = ['ATD3', 'TD3']
    acc_mat = np.zeros((len(method_name_vec), 10, 61))
    for r in range(len(method_name_vec)):

        # file_name_vec = glob.glob('runs/ATD_results2/' + '*' + method_name_vec[r] + '*/test_accuracy.xls')
        for c in range(acc_mat.shape[1]):
            file_name = 'runs/ATD3_results/TD3_' + method_name_vec[r] + '_{}/test_accuracy.xls'.format(c+1)
            print(file_name)
            dfs = pd.read_excel(file_name)
            acc_mat[r, c, :] = dfs.values.astype(np.float)[:, 0]

    max_acc = np.max(acc_mat, axis=-1)
    print('Max acc, mean: {}, std: {}'.format(np.mean(max_acc, axis=-1), np.std(max_acc, axis=-1)))

    for r in range(acc_mat.shape[0]):
        for c in range(acc_mat.shape[1]):
            acc_mat[r, c, :] = smooth(acc_mat[r, c, :], weight=0.8)

    mean_acc = np.mean(acc_mat, axis=1)
    std_acc = np.std(acc_mat, axis=1)
    # kernel = np.ones((1, 1), np.float32) / 1
    # mean_acc = cv2.filter2D(mean_acc, -1, kernel)
    # std_acc = cv2.filter2D(std_acc, -1, kernel)
    t = np.linspace(0, 3, 61)

    fig = plt.figure(figsize=(9, 6))

    plt.tight_layout()
    plt.rcParams.update({'font.size': 15})
    plot_error_line(t, mean_acc, std_acc, legend_vec=['TD3', 'Gait reward + TD3', 'Gait reward + ATD3'])
    # plot_error_line(t, mean_acc, std_acc, legend_vec=['ATD3', 'TD3'])
    # plt.xticks(np.arange(0, 1e5, 5))
    plt.xlabel('Time steps (1e5)')
    plt.ylabel('Average reward')
    plt.savefig('images/test_accuracy.pdf', bbox_inches='tight')
    plt.show()


def plot_joint_angle():
    file_name_vec = glob.glob('video/RoboschoolWalker2d-v1_human_angle_still_steps_ATD3/' + '*0.0.*.xls')
    dfs = pd.read_excel(file_name_vec[0])
    obs_mat = dfs.values.T
    joint_angle_robot = obs_mat[8:20:2, :]

    joint_angle_human = read_table(file_name='../data/joint_angle.xls').T
    for c in range(len(file_name_vec)):
        plot_error_line(joint_angle_robot[0, :], joint_angle_robot[1:3, :], 0 * joint_angle_robot[1:3, :],
                        legend_vec = ['Knee', 'Ankle'])
        plot_error_line(joint_angle_human[0, :], joint_angle_human[1:3, :], 0 * joint_angle_human[1:3, :],
                        legend_vec = ['Knee', 'Ankle'])
        plt.show()


def read_joint_angle_gait(file_name):
    dfs = pd.read_excel(file_name)
    obs_mat = dfs.values.T
    joint_angle_robot = obs_mat[8:20:2, :]
    foot_contact_vec = signal.medfilt(obs_mat[-2, :], 11)
    gait_num_vec = np.zeros(foot_contact_vec.shape)
    pre_idx = 0
    for i in range(1, len(foot_contact_vec)):
        if 1 == foot_contact_vec[i] - foot_contact_vec[i - 1]:
            gait_num_vec[pre_idx:] += 1
            pre_idx = i
    joint_angle_gait = np.zeros((0, 100))
    for gait_num in range(1, int(max(gait_num_vec))):
        if np.sum((gait_num_vec == gait_num).astype(np.int)) > 20:
            joint_angle_resample = signal.resample(joint_angle_robot[:, gait_num_vec == gait_num], 100, axis=1)
            joint_angle_gait = np.r_[joint_angle_gait, joint_angle_resample]
    if 0 == joint_angle_gait.shape[0]:
        return None, None
    joint_angle_gait = joint_angle_gait.reshape((-1, 6, 100))
    return np.mean(joint_angle_gait, axis=0), np.std(joint_angle_gait, axis=0)



def plot_gait():
    joint_angle_mean = np.zeros((4, 6, 100))
    joint_angle_std = np.zeros((4, 6, 100))
    joint_angle_mean[0] = read_table(file_name='../data/joint_angle.xls').T

    method_name_vec = ['', 'human_angle_still_steps', 'human_angle_still_steps_ATD3']

    for i in range(len(method_name_vec)):
        file_name_vec = glob.glob('video/*v1_' + method_name_vec[i] + '/*state.xls')
        joint_angle_mean_mat = np.zeros((0, 6, 100))
        for j in range(len(file_name_vec)):
            print(file_name_vec[j])
            joint_angle_mean_temp, _ = read_joint_angle_gait(file_name_vec[j])
            if joint_angle_mean_temp is not None:
                joint_angle_mean_mat = np.r_[joint_angle_mean_mat, joint_angle_mean_temp.reshape(1, 6, 100)]
                # joint_angle_mean_mat[j,...] = joint_angle_mean_temp
        print(joint_angle_mean_mat.shape[0])
        joint_angle_mean[i+1] = np.mean(joint_angle_mean_mat, axis=0)
        print('Similarity: ', calc_cos_similarity(joint_angle_mean[0],  joint_angle_mean[i+1]))
        joint_angle_std[i+1] = np.std(joint_angle_mean_mat, axis=0)

    joint_angle_mean = joint_angle_mean - joint_angle_mean[..., [0]]
    fig = plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 15})
    t = np.linspace(1, 100, 100)
    y_label_vec = ['Hip angle', 'Knee angle', 'Ankle angle']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plot_error_line(t, joint_angle_mean[:, i, :], joint_angle_std[:, i, :])
        plt.ylabel(y_label_vec[i])

    plt.xlabel('% Gait cycle')
    fig.tight_layout()
    fig.legend(['Human', 'TD3', 'Gait reward + TD3', 'Gait reward + ATD3'],
               loc='lower center', ncol=4, bbox_to_anchor=(0.49, 0.96))
    plt.savefig('images/joint_angle.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()


def plot_Q_value():
    method_name_vec = ['human_angle_still_steps', 'human_angle_still_steps_ATD3']
    reward_Q_list = []
    for i in range(len(method_name_vec)):
        file_name_vec = glob.glob('video/*v1_' + method_name_vec[i] + '/*reward_Q.xls')
        reward_Q_mat = np.zeros((0, 3))
        for j in range(len(file_name_vec)):
        # for j in range(1):
            print(file_name_vec[j])
            dfs = pd.read_excel(file_name_vec[j])
            reward_Q = dfs.values
            reward_Q_mat = np.r_[reward_Q_mat, reward_Q]
        reward_Q_list.append(np.transpose(reward_Q_mat[:,1:]))
        print(reward_Q_list[-1].shape)
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})
    # y_label_vec = ['TD3 Q value', 'ATD3 Q value']
    plt.ylabel('Q_value')
    # plt.subplot(2, 1, i + 1)
    error = np.zeros((2, min(reward_Q_list[0].shape[-1], reward_Q_list[1].shape[-1])))
    print(error.shape)
    for i in range(2):
        error[i, :] = (reward_Q_list[i][0] - reward_Q_list[i][1])[:error.shape[-1]]
    t = np.linspace(1, 100, error.shape[-1])
    plot_error_line(t, error, marker_size=0)
    fig.tight_layout()
    fig.legend(['TD3', 'ATD3'],loc='upper center')
    plt.savefig('images/Q_value.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()


def plot_gait_noise():
    joint_angle_mean = np.zeros((4, 6, 100))
    joint_angle_std = np.zeros((4, 6, 100))
    joint_angle_mean[0] = read_table(file_name='../data/joint_angle.xls').T
    for i in range(3):
        file_name_vec = glob.glob('video/*ATD3/noises/*' + str(0.04 * i) + '.*.xls')
        print(file_name_vec)
        joint_angle_mean[i + 1], joint_angle_std[i + 1] = read_joint_angle_gait(file_name_vec[0])

    joint_angle_mean = joint_angle_mean - joint_angle_mean[..., [0]]

    fig = plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 15})
    t = np.linspace(1, 100, 100)
    y_label_vec = ['Hip angle', 'Knee angle', 'Ankle angle']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plot_error_line(t, joint_angle_mean[:, i, :], joint_angle_std[:, i, :])
        plt.ylabel(y_label_vec[i])

    plt.xlabel('% Gait cycle')
    fig.tight_layout()
    fig.legend(['Human', 'noise = 0.0', 'noise = 0.04', 'noise = 0.08'],
               loc='lower center', ncol=4, bbox_to_anchor=(0.49, 0.96))
    plt.savefig('images/joint_angle_noise.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()

def smooth(scalars, weight = 0.8):
    # Exponential moving average,
    # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.asarray(smoothed)

# # Fig: test acc
# print('------Fig: test acc------')
# plot_test_acc()

# # Fig: joint angle
# print('-----Fig: joint angle-----')
# plot_gait()

# # Fig: joint angle noise
# print('-----Fig: joint angle noise-----')
# plot_gait_noise()


# Fig: Q_value
print('-----Fig: Q value-----')
plot_Q_value()
