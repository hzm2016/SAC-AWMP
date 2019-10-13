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
# sys.path.insert(0,'../')
from code.utils.utils import *
from scipy import stats, signal


def plot_error_line(t, acc_mean_mat, acc_std_mat = None, legend_vec = None,
                    marker_vec=['+', '*', 'o', 'd', 'd', '*', '', '+', 'v', 'x'],
                    line_vec=['--', '-', ':', '-.', '-', '--', '-.', ':', '--', '-.'],
                    marker_size=5,
                    init_idx = 0, idx_step = 1):
    if acc_std_mat is None:
        acc_std_mat = 0 * acc_mean_mat
    # acc_mean_mat, acc_std_mat: rows: methods, cols: time
    color_vec = plt.cm.Dark2(np.arange(8))
    for r in range(acc_mean_mat.shape[0]):
        plt.plot(t, acc_mean_mat[r, :], linestyle=line_vec[idx_step * r + init_idx],
                 marker=marker_vec[idx_step * r + init_idx], markersize=marker_size, linewidth= 2,
                 color=color_vec[(idx_step * r + init_idx) % 8])
        plt.fill_between(t, acc_mean_mat[r, :] - acc_std_mat[r, :],
                         acc_mean_mat[r, :] + acc_std_mat[r, :], alpha=0.2,
                         color=color_vec[(idx_step * r + init_idx) % 8])
    if legend_vec is not None:
        # plt.legend(legend_vec)
        plt.legend(legend_vec, loc = 'upper left')

def plot_reward_curves(reward_name_idx = None, policy_name_vec=None, result_path ='runs/ATD3_walker2d',
                       env_name = 'RoboschoolWalker2d', fig = None):
    if reward_name_idx is None:
        reward_name_idx = [0, 9, 9, 9]
    if policy_name_vec is None:
        policy_name_vec = ['TD3', 'TD3', 'ATD3', 'ATD3_RNN']
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    reward_mat = None
    legend_vec = []
    last_reward = 0.0
    for r in range(len(reward_name_idx)):
        reward_str = connect_str_list(reward_name_vec[:reward_name_idx[r]+1])
        if 0 == reward_name_idx[r]:
            reward_legend = '$r^d$'
        else:
            reward_legend = '$r^d + \hat{r}^g$'
        legend_vec.append(policy_name_vec[r] + ' + ' + reward_legend)
        file_name_vec = glob.glob('{}/*_{}_{}*{}/test_accuracy.xls'.format(
            result_path, policy_name_vec[r], env_name, reward_str))
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # print(file_name)
            dfs = pd.read_excel(file_name)
            acc_vec = dfs.values.astype(np.float)[:, 0]
            if reward_mat is None:
                reward_mat = np.zeros((len(reward_name_idx), len(file_name_vec), len(acc_vec)))
            reward_mat[r, c, :] = acc_vec

        if reward_mat is not None:
            max_acc = np.max(reward_mat[r, :, :], axis=-1)
            # print(max_acc)
            print('Max acc for {} and {}, mean: {}, std: {}, d_reward:{}'.format(
                policy_name_vec[r], reward_str, np.mean(max_acc, axis=-1),
                np.std(max_acc, axis=-1), np.mean(max_acc, axis=-1)-last_reward))
            last_reward = np.mean(max_acc, axis=-1)

    if reward_mat is not None:
        plot_acc_mat(reward_mat, None, env_name, fig=fig)
    return legend_vec


def read_csv_vec(file_name):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    return data[:, -1]

def plot_Q_vals(reward_name_idx = None, policy_name_vec=None, result_path ='runs/ATD3_walker2d',
                       env_name = 'RoboschoolWalker2d'):
    if reward_name_idx is None:
        reward_name_idx = [0, 9, 9, 9]
    if policy_name_vec is None:
        policy_name_vec = ['TD3', 'TD3', 'ATD3', 'ATD3_RNN']
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    Q_val_mat = None
    legend_vec = []
    for r in range(len(policy_name_vec)):
        reward_str = connect_str_list(reward_name_vec[:reward_name_idx[r]+1])
        legend_vec.append(policy_name_vec[r])
        legend_vec.append('True ' + policy_name_vec[r])
        # file_names_list = [glob.glob('{}/*_{}_{}*{}_train-tag*.csv'.format(
        #     result_path, policy_name_vec[r], env_name, reward_str)),
        #     glob.glob('{}/*_{}_{}*{}-tag*.csv'.format(
        #     result_path, policy_name_vec[r], env_name, reward_str))]
        file_names_list = [glob.glob('{}/*_{}_{}*/estimate_Q_vals.xls'.format(
            result_path, policy_name_vec[r], env_name)),
            glob.glob('{}/*_{}_{}*/true_Q_vals.xls'.format(
            result_path, policy_name_vec[r], env_name))]
        for i in range(len(file_names_list)):
        # for file_name_vec in file_names_list:
            file_name_vec = file_names_list[i]
            print(file_name_vec)
            for c in range(len(file_name_vec)):
                file_name = file_name_vec[c]
                dfs = pd.read_excel(file_name)
                Q_vals = dfs.values.astype(np.float)[:, 0]
                # Q_vals = read_csv_vec(file_name)
                if Q_val_mat is None:
                    Q_val_mat = np.zeros((len(reward_name_idx) * 2, len(file_name_vec), 271))
                if Q_vals.shape[0] < Q_val_mat.shape[-1]:
                    Q_vals = np.interp(np.arange(271), np.arange(271, step = 10), Q_vals[:28])
                Q_val_mat[2 * r + i, c, :] = Q_vals[:271]

    if Q_val_mat is not None:
        fig = plt.figure(figsize=(15, 6))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 20})
        plt.subplot(1, 2, 1)
        time_step = Q_val_mat.shape[-1] - 1
        for i in range(Q_val_mat.shape[0]):
            if 0 == i % 2:
                t = np.linspace(0, 0 + 0.01 * time_step, time_step + 1)
                plot_acc_mat(Q_val_mat[[i]],
                             None, env_name, smooth_weight=0.0, plot_std=True,
                             fig_name=None, y_label='Q value', fig = fig, t = t, marker_size = 3, init_idx=i)
            else:
                t = np.linspace(0, 0 + 0.01 * time_step, time_step / 10 + 1)
                plot_acc_mat(Q_val_mat[[i], :, ::10],
                             None, env_name, smooth_weight=0.0, plot_std=True,
                             fig_name=None, y_label='Q value', fig=fig, t=t, init_idx=i, marker_size = 5)
        plt.yticks([0, 50, 100])
        plt.subplot(1, 2, 2)
        Q_val_mat = Q_val_mat[:, :, 90:]
        time_step = Q_val_mat.shape[-1] - 1
        t = np.linspace(1, 1 + 0.01 * time_step, time_step + 1)
        # error_Q_val_mat = (Q_val_mat[[0, 2]] - Q_val_mat[[1, 3]]) / Q_val_mat[[1, 3]]
        error_Q_val_mat = (Q_val_mat[0:6:2] - Q_val_mat[1:6:2]) / np.mean(Q_val_mat[1:6:2],
                                                                          axis = 1, keepdims=True)
        print('Mean absolute normalized error of Q value, TD3: {}, ATD3: {}, ATD3_RNN: {}'.format(
            np.mean(np.abs(error_Q_val_mat[0, :, -50:])), np.mean(np.abs(error_Q_val_mat[1, :, -50:])),
            np.mean(np.abs(error_Q_val_mat[2, :, -50:]))))
        plot_acc_mat(error_Q_val_mat,
                     None, env_name, smooth_weight=0.0, plot_std=True,
                     fig_name=None, y_label='Error of Q value / True Q value',
                     fig = fig, t = t, init_idx=0, idx_step=2, marker_size = 3)
        plt.yticks([0, 1, 2])
        legend = fig.legend(legend_vec,
                            loc='lower center', ncol=3, bbox_to_anchor=(0.48, 0.93),
                            frameon=False)
        plt.savefig('images/{}_{}.pdf'.format(env_name, 'Q_value'), bbox_inches='tight')
        fig.tight_layout()
        plt.show()



def plot_ablation_reward(result_path ='runs/ATD3_walker2d',
                         env_name = 'RoboschoolWalker2d'):
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    reward_tick_vec = ['r^d', 'r^s', 'r^n', 'r^{lhs}', 'r^{cg}', 'r^{gs}', 'r^{fr}', 'r^{f}', 'r^{gv}', 'r^{po}']
    # reward_name_vec = ['r_d', 'r_s', 'r_f', 'r_n', 'r_gv', 'r_lhs', 'r_gs', 'r_cg', 'r_fr', 'r_po']
    # reward_tick_vec = ['r^d', 'r^s', 'r^f', 'r^n', 'r^{gv}', 'r^{lhs}', 'r^{gs}', 'r^{cg}', 'r^{fr}', 'r^{po}']

    acc_mat = None
    x_tick_vec = []
    for r in range(len(reward_name_vec)):
        reward_str = connect_str_list(reward_name_vec[:r+1])
        # reward_tick = '$r^{}$ = $r^d$'.format(r)
        if 0 == r:
            reward_tick = '${}$'.format(reward_tick_vec[r])
        else:
            reward_tick = '$+{}$'.format(reward_tick_vec[r])
        x_tick_vec.append(reward_tick)
        file_name_vec = glob.glob('{}/*_{}_{}*{}/test_accuracy.xls'.format(
            result_path, 'TD3', env_name, reward_str))
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # print(file_name)
            dfs = pd.read_excel(file_name)
            acc_vec = dfs.values.astype(np.float)[:, 0]
            if acc_mat is None:
                acc_mat = np.zeros((len(reward_name_vec), len(file_name_vec), len(acc_vec)))
            acc_mat[r, c, :] = acc_vec

    if acc_mat is not None:
        # plot_acc_mat(acc_mat, x_tick_vec, 'ablation_study', plot_std=False)
        max_acc_mat = np.max(acc_mat, axis=-1)
        print(np.mean(max_acc_mat, axis=-1))
        plot_error_bar(np.arange(10), max_acc_mat, x_tick_vec)


def plot_error_bar(x_vec, y_mat, x_tick_vec = None):
    mean_vec = np.mean(y_mat, axis = -1)
    std_vec = np.std(y_mat, axis = -1)
    len_vec = len(x_vec)
    fig = plt.figure(figsize=(9, 3))
    plt.tight_layout()
    plt.rcParams.update({'font.size': 15})

    plt.errorbar(x_vec, mean_vec, yerr = std_vec, fmt='-', elinewidth= 1,
                 solid_capstyle='projecting', capsize= 3, color = 'black')
    plt.ylabel('Average reward')
    if x_tick_vec is not None:
        plt.xticks(np.arange(len(x_tick_vec)), x_tick_vec)
    plt.savefig('images/ablation_reward.pdf', bbox_inches='tight')
    plt.show()


def connect_str_list(str_list):
    if 0 >= len(str_list):
        return ''
    str_out = str_list[0]
    for i in range(1, len(str_list)):
        str_out = str_out + '_' + str_list[i]
    return str_out


def plot_acc_mat(acc_mat, legend_vec, env_name, plot_std = True, smooth_weight = 0.8, eval_freq = 0.05,
                 t = None, fig = None, fig_name = None, y_label = 'Test reward',
                 init_idx = 0, idx_step = 1, marker_size = 2):
    # print(legend_vec)
    for r in range(acc_mat.shape[0]):
        for c in range(acc_mat.shape[1]):
            acc_mat[r, c, :] = smooth(acc_mat[r, c, :], weight=smooth_weight)
    mean_acc = np.mean(acc_mat, axis=1)
    std_acc = np.std(acc_mat, axis=1)
    # kernel = np.ones((1, 1), np.float32) / 1
    # mean_acc = cv2.filter2D(mean_acc, -1, kernel)
    # std_acc = cv2.filter2D(std_acc, -1, kernel)
    if t is None:
        time_step = acc_mat.shape[-1] - 1
        t = np.linspace(0, eval_freq * time_step, time_step+1)
    if fig is None:
        fig = plt.figure(figsize=(9, 6))
        # fig = plt.figure()
        plt.tight_layout()
        plt.rcParams.update({'font.size': 15})
    if plot_std:
        plot_error_line(t, mean_acc, std_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step = idx_step, marker_size=marker_size)
    else:
        plot_error_line(t, mean_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step = idx_step, marker_size=marker_size)
    plt.xlabel('{}: '.format(env_name) + r'Time steps ($1 \times 10^{5}$)')
    plt.xlim((min(t), max(t)))
    plt.ylabel(y_label)
    if fig_name is not None:
        plt.savefig('images/{}_{}.pdf'.format(env_name, fig_name), bbox_inches='tight')
    if fig is None:
        plt.show()

def plot_test_acc():
    method_name_vec = ['','human_angle_still_steps', 'human_angle_still_steps_ATD3']
    # method_name_vec = ['ATD3', 'TD3']
    acc_mat = np.zeros((len(method_name_vec), 10, 61))
    for r in range(len(method_name_vec)):
        # file_name_vec = glob.glob('runs/ATD_results2/' + '*' + method_name_vec[r] + '*/test_accuracy.xls')
        for c in range(acc_mat.shape[1]):
            file_name = 'runs/ATD3_walker2d/TD3_' + method_name_vec[r] + '_{}/test_accuracy.xls'.format(c+1)
            print(file_name)
            dfs = pd.read_excel(file_name)
            acc_mat[r, c, :] = dfs.values.astype(np.float)[:, 0]

    max_acc = np.max(acc_mat, axis=-1)
    print(max_acc)
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
    plot_error_line(t, mean_acc, std_acc, legend_vec=['TD3', 'Gait reward + TD3', 'Gait reward + ATD3'], init_idx=1)
    # plot_error_line(t, mean_acc, std_acc, legend_vec=['ATD3', 'TD3'])
    # plt.xticks(np.arange(0, 1e5, 5))
    #r'Time steps (1 x 10^5)'
    plt.xlabel(r'Time steps ($1 \times 10^{5}$)')
    plt.xlim((min(t), max(t)))
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
    joint_idx_list = [[8, 10, 12], [14, 16, 18]]
    joint_angle_gait = np.zeros((0, 100))
    for r in range(2):
        joint_angle_robot = np.copy(obs_mat[joint_idx_list[r], :])
        foot_contact_vec = signal.medfilt(obs_mat[-2 + r, :], 11)
        gait_num_vec = np.zeros(foot_contact_vec.shape)
        pre_idx = 0
        for i in range(1, len(foot_contact_vec)):
            if 1 == foot_contact_vec[i] - foot_contact_vec[i - 1]:
                gait_num_vec[pre_idx:] += 1
                pre_idx = i
        for gait_num in range(1, int(max(gait_num_vec))):
            if np.sum((gait_num_vec == gait_num).astype(np.int)) > 25:
                joint_angle_resample = signal.resample(joint_angle_robot[:, gait_num_vec == gait_num], 100, axis=1)
                joint_angle_gait = np.r_[joint_angle_gait, joint_angle_resample]

    if 0 == joint_angle_gait.shape[0]:
        return None, None
    joint_angle_gait = joint_angle_gait.reshape((-1, 3, 100))
    return np.mean(joint_angle_gait, axis=0), np.std(joint_angle_gait, axis=0)


def plot_all_test_reward():
    fig = plt.figure(figsize=(15, 6))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 20})
    plt.subplot(1, 2, 1)
    legend_vec = plot_reward_curves(result_path='runs/ATD3_walker2d',
                       env_name='RoboschoolWalker2d',
                       policy_name_vec=['TD3', 'TD3', 'ATD3', 'ATD3_RNN'],
                       reward_name_idx=[0, 4, 4, 4], fig = fig)
    plt.yticks([0, 1000, 2000])
    plt.xticks([0, 1.5, 3])
    plt.subplot(1, 2, 2)
    plot_reward_curves(result_path='runs/ATD3_Atlas', env_name='WebotsAtlas',
                       policy_name_vec=['TD3', 'TD3', 'ATD3', 'ATD3_RNN'],
                       reward_name_idx=[0, 4, 4, 4], fig = fig)
    plt.yticks([0, 1000, 2000, 3000])
    plt.xticks([0, 5, 10])
    # legend_vec = plot_gait_angle(env_name='RoboschoolWalker2d', gait_name='run', plot_col = 1)
    # plot_gait_angle(env_name='WebotsAtlas', gait_name='run', plot_col = 2)

    print(legend_vec)
    legend = fig.legend(legend_vec,
               loc='lower center', ncol=4, bbox_to_anchor=(0.50, 0.93), frameon=False)
    fig.tight_layout()
    # legend.get_frame().set_facecolor('none')
    plt.savefig('images/test_reward.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()


def plot_all_gait_angle():
    fig = plt.figure(figsize=(15, 10))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 20})

    legend_vec = plot_gait_angle(env_name='RoboschoolWalker2d', gait_name='run', plot_col = 1)
    plot_gait_angle(env_name='WebotsAtlas', gait_name='run', plot_col = 2)

    print(legend_vec)
    legend = fig.legend(legend_vec,
               loc='lower center', ncol=3, bbox_to_anchor=(0.48, 0.93), frameon=False)
    fig.tight_layout()
    # legend.get_frame().set_facecolor('none')
    plt.savefig('images/joint_angle.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()


def plot_gait_angle(reward_name_idx = None, policy_name_vec=None, result_path ='video',
                       env_name = 'RoboschoolWalker2d', gait_name = 'run', plot_col = 1):
    if reward_name_idx is None:
        reward_name_idx = [0, 4, 4, 4]
    if policy_name_vec is None:
        policy_name_vec = ['TD3','TD3', 'ATD3', 'ATD3_RNN']
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    legend_vec = []

    joint_angle_mean = np.zeros((len(reward_name_idx)+1, 3, 100))
    joint_angle_mean[-1] = read_table(file_name='../data/joint_angle.xls', sheet_name=gait_name).T[0:3]
    joint_angle_std = np.zeros((len(reward_name_idx)+1, 3, 100))
    for r in range(len(reward_name_idx)):
        reward_str = connect_str_list(reward_name_vec[:reward_name_idx[r]+1])
        if 0 == reward_name_idx[r]:
            reward_legend = '$r^d$'
        else:
            reward_legend = '$r^d + \hat{r}^g$'
        legend_vec.append(policy_name_vec[r] + ' + ' + reward_legend)
        file_name_vec = glob.glob('{}/{}*_{}_{}/*.xls'.format(
            result_path, env_name, policy_name_vec[r], reward_str))
        print(file_name_vec)
        joint_angle_mean_mat = np.zeros((0, 3, 100))
        for j in range(len(file_name_vec)):
            print(file_name_vec[j])
            joint_angle_mean_temp, _ = read_joint_angle_gait(file_name_vec[j])
            if joint_angle_mean_temp is not None:
                joint_angle_mean_mat = np.r_[joint_angle_mean_mat, joint_angle_mean_temp.reshape(1, 3, 100)]
                # joint_angle_mean_mat[j,...] = joint_angle_mean_temp
        joint_angle_mean[r] = np.mean(joint_angle_mean_mat, axis=0)
        joint_angle_std[r] = np.std(joint_angle_mean_mat, axis=0)

    joint_angle_mean = joint_angle_mean - joint_angle_mean[..., [0]]
    for r in range(joint_angle_mean.shape[0] - 1):
        print('Similarity of {}: '.format(legend_vec[r]),
              calc_cos_similarity(joint_angle_mean[-1], joint_angle_mean[r]))

    t = np.linspace(0, 100, 100)
    y_label_vec = ['Hip angle', 'Knee angle', 'Ankle angle']
    print(joint_angle_mean.shape)
    for i in range(3):
        plt.subplot(3, 2,  2*i + plot_col)
        plot_error_line(t, joint_angle_mean[:, i, :])
        plt.xlim((min(t), max(t)))
        plt.xticks([min(t), (min(t) + max(t)) / 2, max(t)])
        plt.ylim((-1.1, 1.1))
        plt.yticks([-1, 0, 1])
        # if 1 == plot_col:
        plt.ylabel(y_label_vec[i])

    plt.xlabel('{}: Gait cycle (%)'.format(env_name))
    legend_vec.append('Human')
    return legend_vec



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
    t = np.linspace(0, 100, 100)
    y_label_vec = ['Hip angle', 'Knee angle', 'Ankle angle']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plot_error_line(t, joint_angle_mean[:, i, :], joint_angle_std[:, i, :])
        plt.xlim((min(t), max(t)))
        plt.ylabel(y_label_vec[i])

    plt.xlabel('\% Gait cycle')
    fig.tight_layout()
    fig.legend(['Human', 'TD3', 'Gait reward + TD3', 'Gait reward + ATD3'],
               loc='lower center', ncol=4, bbox_to_anchor=(0.49, 0.96))
    plt.savefig('images/joint_angle.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()

def calc_TD_reward(reward_Q):
    reward_Q_TD = np.zeros(reward_Q.shape)
    reward_Q_TD[:, 0] = reward_Q[:, 0]
    for r in range(reward_Q.shape[0]-1):
        reward_Q_TD[r,1:3] = reward_Q[r, 1:3] - 0.99 * np.min(reward_Q[r+1, 1:3])
    return reward_Q_TD


def calc_expected_reward(reward_Q):
    reward = np.copy(reward_Q[:, 0])
    r = 0
    # for r in range(reward_Q.shape[0]-1):
    for c in range(r+1, reward_Q.shape[0]):
        reward[r] += 0.99 ** (c-r) * reward[c]
        # reward[r] += np.min(0.99 * reward_Q[r + 1, 1:3])
    init_rewar_Q = reward_Q[[0],:]
    init_rewar_Q[0, 0] = reward[0]
    return init_rewar_Q

def plot_gait_noise():
    joint_angle_mean = np.zeros((4, 6, 100))
    joint_angle_std = np.zeros((4, 6, 100))
    joint_angle_mean[0] = read_table(file_name='../data/joint_angle.xls').T
    for i in range(3):
        file_name_vec = glob.glob('video/*ATD3/noises/*' + str(0.04 * i) + '*state.xls')
        print(file_name_vec)
        joint_angle_mean[i + 1], joint_angle_std[i + 1] = read_joint_angle_gait(file_name_vec[0])

    joint_angle_mean = joint_angle_mean - joint_angle_mean[..., [0]]

    fig = plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 15})
    t = np.linspace(0, 100, 100)
    y_label_vec = ['Hip angle', 'Knee angle', 'Ankle angle']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plot_error_line(t, joint_angle_mean[:, i, :], joint_angle_std[:, i, :])
        plt.ylabel(y_label_vec[i])
        plt.xlim((min(t), max(t)))
    plt.xlabel('\% Gait cycle')
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

# # Fig: ablation study for different rewards
# print('------Fig: ablation reward------')
# plot_ablation_reward()

# Fig: test acc
print('------Fig: test reward------')
# plot_all_test_reward()
# plot_reward_curves(result_path = 'runs/ATD3_walker2d',
#                 env_name = 'RoboschoolWalker2d',
#                 policy_name_vec = ['TD3', 'TD3', 'ATD3', 'ATD3_RNN'],
#                 reward_name_idx = [0, 4, 4, 4])
# plot_reward_curves(result_path ='runs/ATD3_Atlas', env_name ='WebotsAtlas',
#                    policy_name_vec=['TD3', 'TD3', 'ATD3', 'ATD3_RNN'],
#                    reward_name_idx=[0, 4, 4, 4])

## Fig: Q-value
# print('------Fig: Q value ------')
plot_Q_vals(result_path = 'runs/ATD3_walker2d_Q_value',
            env_name = 'RoboschoolWalker2d',
            policy_name_vec = ['TD3', 'ATD3', 'ATD3_RNN'],
            reward_name_idx = [0, 0, 0])


# # Fig: joint angle
# print('-----Fig: joint angle-----')
# plot_all_gait_angle()

# # Fig: Q_value
# print('-----Fig: Q value-----')
# plot_Q_value()
