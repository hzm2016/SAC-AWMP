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
from scipy import stats, signal


def plot_error_line(t, acc_mean_mat, acc_std_mat, legend_vec,
                    marker_vec=['o', '+', 'v', 'x', 'd', '*', ''],
                    line_vec=['-', '--', '-.', ':', '-', '--', '-.'],
                    line_width_vec=[2, 2, 2, 2, 2, 2, 2], marker_size=5
                    ):
    # acc_mean_mat, acc_std_mat: rows: methods, cols: time
    color_vec = plt.cm.Dark2(np.arange(4))
    for r in range(acc_mean_mat.shape[0]):
        plt.plot(t, acc_mean_mat[r, :], linestyle=line_vec[r],
                 marker=marker_vec[r], markersize=marker_size, linewidth=line_width_vec[r],
                 color=color_vec[r])
        plt.fill_between(t, acc_mean_mat[r, :] - acc_std_mat[r, :],
                         acc_mean_mat[r, :] + acc_std_mat[r, :], alpha=0.1, color=color_vec[r])
    plt.legend(legend_vec)


def plot_test_acc():
    # method_name_vec = ['', 'human_angle_still_steps', 'human_angle_still_steps_ATD3']
    method_name_vec = ['ATD3', 'TD3']
    acc_mat = np.zeros((len(method_name_vec), 10, 61))
    for r in range(len(method_name_vec)):
        # file_name_vec = glob.glob('runs/ATD3_results/' + '*v1_' + method_name_vec[r] + '/test_accuracy.xls')
        file_name_vec = glob.glob('runs/ATD_results2/' + '*' + method_name_vec[r] + '*/test_accuracy.xls')
        for c in range(acc_mat.shape[1]):
            dfs = pd.read_excel(file_name_vec[c])
            acc_mat[r, c, :] = smooth(dfs.values.astype(np.float)[:, 0])


    mean_acc = np.mean(acc_mat, axis=1)
    std_acc = np.std(acc_mat, axis=1)
    # kernel = np.ones((1, 1), np.float32) / 1
    # mean_acc = cv2.filter2D(mean_acc, -1, kernel)
    # std_acc = cv2.filter2D(std_acc, -1, kernel)
    t = np.linspace(0, 3, 61)

    fig = plt.figure(figsize=(9, 6))
    plt.tight_layout()
    plt.rcParams.update({'font.size': 15})
    # plot_error_line(t, mean_acc, std_acc, legend_vec=['TD3', 'Gait reward + TD3', 'Gait reward + ATD3'])
    plot_error_line(t, mean_acc, std_acc, legend_vec=['ATD3', 'TD3'])
    # plt.xticks(np.arange(0, 1e5, 5))
    plt.xlabel('Time steps (1e5)')
    plt.ylabel('Average reward')
    plt.savefig('images/test_accuracy.pdf', bbox_inches='tight')
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

# Fig: test acc
plot_test_acc()

