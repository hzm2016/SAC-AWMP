# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:29:32 2019

@author: kuangen
"""
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import pandas as pd
import openpyxl
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
                 marker=marker_vec[idx_step * r + init_idx], markersize=marker_size, linewidth= 2.0,
                 color=color_vec[(idx_step * r + init_idx) % 8])
        plt.fill_between(t, acc_mean_mat[r, :] - acc_std_mat[r, :],
                         acc_mean_mat[r, :] + acc_std_mat[r, :], alpha=0.2,
                         color=color_vec[(idx_step * r + init_idx) % 8])
    if legend_vec is not None:
        # plt.legend(legend_vec)
        plt.legend(legend_vec, loc='upper left')


def read_csv_vec(file_name):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    return data[:, -1]


def write_matrix_to_xlsx(data_mat, file_path = 'data/state_of_art_test_reward.xlsx', env_name = 'Ant',
                         index_label = ['DDPG']):
    df = pd.DataFrame(data_mat)
    writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a')
    df.to_excel(writer, sheet_name=env_name, index_label=tuple(index_label), header=False)
    writer.save()
    writer.close()

def write_to_existing_table(data, file_name, sheet_name = 'label'):
    xl = pd.read_excel(file_name, sheet_name=None, header=0, index_col=0, dtype='object')
    xl[sheet_name].iloc[1:, :5] = data
    xl[sheet_name].iloc[1:, 5] = np.mean(data, axis=-1)
    xl[sheet_name].iloc[0, 5] = np.max(xl[sheet_name].iloc[1:, 5])
    xl[sheet_name].iloc[1:, 6] = np.std(data, axis=-1)
    xl[sheet_name].iloc[0, 6] = np.min(xl[sheet_name].iloc[1:, 6])
    print(xl[sheet_name])
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        for ws_name, df_sheet in xl.items():
            df_sheet.to_excel(writer, sheet_name=ws_name)

def plot_ablation_reward(result_path ='runs/ATD3_walker2d',
                         env_name = 'RoboschoolWalker2d'):
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    reward_tick_vec = ['r^d', 'r^s', 'r^n', 'r^{lhs}', 'r^{cg}', 'r^{gs}', 'r^{fr}', 'r^{f}', 'r^{gv}', 'r^{po}']
    # reward_name_vec = ['r_d', 'r_s', 'r_f', 'r_n', 'r_gv', 'r_lhs', 'r_gs', 'r_cg', 'r_fr', 'r_po']
    # reward_tick_vec = ['r^d', 'r^s', 'r^f', 'r^n', 'r^{gv}', 'r^{lhs}', 'r^{gs}', 'r^{cg}', 'r^{fr}', 'r^{po}']

    acc_mat = None
    x_tick_vec = []
    len_reward = 6
    for r in range(len_reward):
        reward_str = connect_str_list(reward_name_vec[:r+1])
        # reward_tick = '$r^{}$ = $r^d$'.format(r)
        if 0 == r:
            reward_tick = '${}$'.format(reward_tick_vec[r])
        else:
            reward_tick = '$+{}$'.format(reward_tick_vec[r])
        x_tick_vec.append(reward_tick)
        file_name_vec = glob.glob('{}/*_{}_{}*{}/test_accuracy_old.xls'.format(
            result_path, 'TD3', env_name, reward_str))
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # print(file_name)
            dfs = pd.read_excel(file_name)
            acc_vec = dfs.values.astype(np.float)[:, 0]
            if acc_mat is None:
                acc_mat = np.zeros((len_reward, len(file_name_vec), len(acc_vec)))
            acc_mat[r, c, :] = acc_vec

    if acc_mat is not None:
        # plot_acc_mat(acc_mat, x_tick_vec, 'ablation_study', plot_std=False)
        max_acc_mat = np.max(acc_mat, axis=-1)
        print(np.mean(max_acc_mat, axis=-1))
        plot_error_bar(np.arange(len_reward), max_acc_mat, x_tick_vec)


def plot_error_bar(x_vec, y_mat, x_tick_vec = None):
    mean_vec = np.mean(y_mat, axis = -1)
    std_vec = np.std(y_mat, axis = -1)
    len_vec = len(x_vec)
    fig = plt.figure(figsize=(3.5, 1))
    plt.tight_layout()
    plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})

    plt.errorbar(x_vec, mean_vec, yerr = std_vec, fmt='-', elinewidth= 1,
                 solid_capstyle='projecting', capsize= 3, color = 'black')
    plt.ylabel('Test reward')
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


def plot_all_test_reward():
    fig = plt.figure(figsize=(3.5 , 2.5))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})
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
               loc='lower center', ncol=2, bbox_to_anchor=(0.50, 0.90), frameon=False)
    fig.tight_layout()
    # legend.get_frame().set_facecolor('none')
    plt.savefig('images/test_reward.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()


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

    plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})
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


def plot_acc_mat(acc_mat,
                 legend_vec, env_name,
                 plot_std=True, smooth_weight=0.8, eval_freq=0.05,
                 t = None, fig=None, fig_name=None,
                 y_label='Test reward',
                 init_idx=0,
                 idx_step=1,
                 marker_size=2):
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
        plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})
    
    if plot_std:
        plot_error_line(t, mean_acc, std_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step=idx_step, marker_size=marker_size)
    else:
        plot_error_line(t, mean_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step=idx_step, marker_size=marker_size)

    plt.xlabel('Time steps ' + r'($1 \times 10^{5}$)' + '\n{}'.format(env_name))
    plt.xlim((min(t), max(t)))
    plt.ylabel(y_label, fontsize=10)
    if fig is None:
        plt.show()


def plot_roboschool_test_reward():
    env_name_vec = [
        # 'RoboschoolHopper-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolAnt-v1',
        'Walker2d-v2',
        'Ant-v2',
        'Hopper-v2',
        # 'HalfCheetah-v2',
    ]
    
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 10, 'font.serif': 'Times New Roman'})
    
    option_num = ['td3']
    # policy_name_vec = ['HRLSAC', 'SAC', 'TD3']
    policy_name_vec = ['DDPG', 'SAC', 'TD3']
    
    # for i in range(len(option_num)):
    #     plt.subplot(2, 2, i+1)
    #     legend_vec = plot_reward_curves(result_path='runs',
    #                                     option_num=option_num[i],
    #                                     env_name=env_name_vec[0],
    #                                     policy_name_vec=policy_name_vec,
    #                                     fig=fig)
    #     print(legend_vec)
    for i in range(len(env_name_vec)):
        plt.subplot(2, 2, i + 1)
        legend_vec = plot_reward_curves(result_path='runs',
                                        option_num=option_num[0],
                                        env_name=env_name_vec[i],
                                        policy_name_vec=policy_name_vec,
                                        fig=fig)
        print(legend_vec)
        # plt.yticks([0, 1000, 2000, 3000])
        # plt.xticks([0, 5, 10])
    
    legend_labels = ['SAC-AWMP', 'SAC', 'TD3']
    legend = fig.legend(legend_labels,
                        loc='lower center',
                        ncol=len(legend_labels),
                        bbox_to_anchor=(0.50, 0.96),
                        frameon=False)
    fig.tight_layout()
    plt.savefig('./figure/evaluation_reward_option_4_four_env.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_reward_curves(policy_name_vec=None,
                       result_path='runs/ATD3_walker2d',
                       option_num='option_8',
                       env_name='RoboschoolWalker2d',
                       fig=None,
                       fig_name='test_reward',
                       smooth_weight=0.8,
                       eval_freq=0.05):
    reward_mat = None
    legend_vec = []
    last_reward = 0.0
    for r in range(len(policy_name_vec)):
        legend_vec.append(policy_name_vec[r])
        file_name_vec = glob.glob('{}/{}/{}_{}*/test_accuracy.npy'.format(
            result_path, 'mujoco_1e6_'+option_num, policy_name_vec[r], env_name))
        print(file_name_vec)
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # dfs = pd.read_excel(file_name)
            dfs = np.load(file_name)
            # acc_vec = dfs.values.astype(np.float)[:, 0]
            acc_vec = dfs
            print('data_length', len(acc_vec))
            data_length = 101
            if reward_mat is None:
                reward_mat = np.zeros((len(policy_name_vec), len(file_name_vec), data_length))
            reward_mat[r, c, :] = acc_vec[:data_length]

        if reward_mat is not None:
            max_acc = np.max(reward_mat[r, :, :], axis=-1)
            # print(max_acc)
            print('Max acc for {}, mean: {}, std: {}, d_reward:{}'.format(
                policy_name_vec[r], np.mean(max_acc, axis=-1),
                np.std(max_acc, axis=-1), np.mean(max_acc, axis=-1)-last_reward))
            last_reward = np.mean(max_acc, axis=-1)

    if reward_mat is not None:
        plot_acc_mat(reward_mat, None, env_name, fig=fig, fig_name=fig_name,
                     smooth_weight=smooth_weight, eval_freq=eval_freq, marker_size=0)
    return legend_vec


def plot_curve_one_fig(policy_name_vec=None,
                       result_path='runs/ATD3_walker2d',
                       option_num='option_8',
                       env_name='RoboschoolWalker2d',
                       fig=None,
                       fig_name='test_reward',
                       smooth_weight=0.8,
                       eval_freq=0.05):
    reward_mat = None
    legend_vec = []
    last_reward = 0.0
    index = 0
    print(len(option_num))
    for r in range(len(option_num)):
        # legend_vec.append(policy_name_vec[r])
        file_name_vec = glob.glob('{}/{}/{}_{}*/test_accuracy.npy'.format(
            result_path, 'mujoco_1e6_' + option_num[r], policy_name_vec, env_name))
        print('r', r)
        print(file_name_vec)
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # dfs = pd.read_excel(file_name)
            dfs = np.load(file_name)
            # acc_vec = dfs.values.astype(np.float)[:, 0]
            acc_vec = dfs
            print('data_length', len(acc_vec))
            data_length = 101
            if reward_mat is None:
                reward_mat = np.zeros((len(option_num), len(file_name_vec), data_length))
            reward_mat[r, c, :] = acc_vec[:data_length]

        # if reward_mat is not None:
        #     max_acc = np.max(reward_mat[r, :, :], axis=-1)
        #     # print(max_acc)
        #     print('Max acc for {}, mean: {}, std: {}, d_reward:{}'.format(
        #         policy_name_vec, np.mean(max_acc, axis=-1),
        #         np.std(max_acc, axis=-1), np.mean(max_acc, axis=-1)-last_reward))
        #     last_reward = np.mean(max_acc, axis=-1)
    print(reward_mat.shape)
    if reward_mat is not None:
        plot_acc_mat(reward_mat, None, env_name, fig=fig, fig_name=fig_name,
                     smooth_weight=smooth_weight, eval_freq=eval_freq, marker_size=0)
    return legend_vec


def plot_ablation_study():
    env_name_vec = [
        # 'RoboschoolHopper-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolAnt-v1',
        # 'Walker2d-v2',
        # 'Ant-v2',
        'Hopper-v2',
        # 'HalfCheetah-v2',
    ]
    
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 10, 'font.serif': 'Times New Roman'})
    
    option_num = ['option_2',
                  'option_4',
                  'option_8']
    policy_name_vec = ['HRLSAC', 'SAC']
    legend_labels = [['SAC-AWMP($\mathcal{O}=2$)', 'SAC'],
                     ['SAC-AWMP($\mathcal{O}=4$)', 'SAC'],
                     ['SAC-AWMP($\mathcal{O}=8$)', 'SAC']]
    legend_option_num = ['$\mathcal{O} = 2$', '$\mathcal{O} = 4$', '$\mathcal{O} = 8$']
    # option_num = ['option_8_target_tau_001',
    #               'option_8_target_tau_0001',
    #               'option_8_target_tau_00001']
    
    # policy_name_vec = ['HRLSAC_Target']
    
    for i in range(len(option_num)):
        plt.subplot(2, 2, i+1)
        plot_reward_curves(result_path='runs',
                           option_num=option_num[i],
                           env_name=env_name_vec[0],
                           policy_name_vec=policy_name_vec,
                           fig=fig)
        plt.legend(legend_labels[i],
                   loc='lower center',
                   ncol=len(legend_labels[i]),
                   bbox_to_anchor=(0.4, 0.96),
                   frameon=False)
    plt.subplot(224)
    plot_curve_one_fig(result_path='runs',
                       option_num=option_num,
                       env_name=env_name_vec[0],
                       policy_name_vec='HRLSAC',
                       fig=fig)
    plt.title('SAC-AWMP', fontsize=10)
    plt.legend(legend_option_num)

    # for i in range(len(env_name_vec)):
    #     plt.subplot(2, 2, i + 1)
    #     legend_vec = plot_reward_curves(result_path='runs',
    #                                     option_num=option_num[0],
    #                                     env_name=env_name_vec[i],
    #                                     policy_name_vec=policy_name_vec,
    #                                     fig=fig)
    
    # plt.yticks([0, 1000, 2000, 3000])
    # plt.xticks([0, 5, 10])
    # legend = fig.legend(policy_name_vec,
    #                     loc='lower center', ncol=len(policy_name_vec),
    #                     bbox_to_anchor=(0.50, 0.96), frameon=False)
    fig.tight_layout()
    plt.savefig('./figure/evaluation_reward_hopper_2d.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_target_policy():
    env_name_vec = [
        # 'RoboschoolHopper-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolAnt-v1',
        'Walker2d-v2',
        # 'Ant-v2',
        # 'Hopper-v2',
        # 'HalfCheetah-v2',
    ]
    
    fig = plt.figure(figsize=(6, 2.5))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 10, 'font.serif': 'Times New Roman'})
    
    # option_num = ['option_8_target_tau_001',
    #               'option_8_target_tau_0001',
    #               'option_8_target_tau_00001']
    
    option_num = ['option_8_target_tau_00001']
    
    lengend_labels = [r'$\tau_Q = 0.1$', r'$\tau_Q = 0.001$', r'$\tau_Q = 0.0001$']
    policy_name_vec = ['HRLSAC_Target', 'TD3']
    
    # for i in range(len(option_num)):
    #     plt.subplot(2, 2, i + 1)
    #     plot_reward_curves(result_path='runs',
    #                        option_num=option_num[i],
    #                        env_name=env_name_vec[0],
    #                        policy_name_vec=policy_name_vec,
    #                        fig=fig)
    #     plt.legend(policy_name_vec,
    #                loc='lower center', ncol=len(policy_name_vec),
    #                bbox_to_anchor=(0.50, 0.96), frameon=False)
    ax_1 = plt.subplot(1, 2, 1)
    plot_curve_one_fig(result_path='runs',
                       option_num=option_num,
                       env_name=env_name_vec[0],
                       policy_name_vec='HRLSAC_Target',
                       fig=fig)
    
    # ax_2 = plt.subplot(1, 2, 2)
    # plot_curve_one_fig(result_path='runs',
    #                    option_num=option_num,
    #                    env_name=env_name_vec[1],
    #                    policy_name_vec='HRLSAC_Target',
    #                    fig=fig)

    # fig.legend([ax_1, ax_2],
    #            labels=lengend_labels,
    #            loc='upperr',
    #            ncol=len(lengend_labels),
    #            bbox_to_anchor=(0.4, 0.55, 0.5, 0.5),
    #            frameon=False)

    
    # plt.yticks([0, 1000, 2000, 3000])
    # plt.xticks([0, 5, 10])
    fig.tight_layout()
    plt.savefig('./figure/evaluation_reward_hopper_different_target.pdf',
                bbox_inches='tight',
                pad_inches=0.1)
    plt.show()
    
    
# # # Fig: ablation study for different rewards
# print('------Fig: ablation reward------')
# plot_ablation_study()
# plot_target_policy()
#
# # Fig: test acc
# print('------Fig: test reward------')
# plot_all_test_reward()
#
# ## Fig: Q-value
# print('------Fig: Q value ------')
# plot_Q_vals(result_path = 'runs/ATD3_walker2d_Q_value',
#             env_name = 'RoboschoolWalker2d',
#             policy_name_vec = ['TD3', 'ATD3', 'ATD3_RNN'],
#             reward_name_idx = [0, 0, 0])
#
#
# # # # Fig: joint angle
# print('-----Fig: joint angle-----')
# plot_all_gait_angle()


# Fig: test acc
# print('------Fig: test reward------')
plot_roboschool_test_reward()
# plot_ablation_study()
# plot_target_policy()

# new_data = np.zeros(202)
# data = np.load('./runs/mujoco_1e6_option_8/HRLSAC_Ant-v2_seed_1/test_accuracy.npy')
# new_data[:168] = data[:168]
# new_data[168:202] = data[168-202:]
# np.save('./runs/mujoco_1e6_option_8/HRLSAC_Ant-v2_seed_1/test_accuracy.npy', new_data)
# print(data.shape)