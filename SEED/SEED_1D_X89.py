##############
## SEED数据集提取每个通道5个频段的DE特征，
## 并将62个通道转化为8*9*5的三维输入，其中8*9表示62个通道转化后的二维平面，5表示5种频段
##############

import os
import sys
import math
import numpy as np
# import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat


def decompose(file, name):
    # trial*channel*sample
    data = loadmat(file)
    frequency = 200

    decomposed_de = np.empty([0, 62, 5])
    label = np.array([])
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(15):
        # tmp_idx = trial + 3
        # tmp_name = list(data.keys())[tmp_idx]
        tmp_trial_signal = data[name + '_eeg' + str(trial + 1)]
        num_sample = int(len(tmp_trial_signal[0]) / 100)
        print('{}-{}'.format(trial + 1, num_sample))

        temp_de = np.empty([0, num_sample])
        label = np.append(label, [all_label[trial]] * num_sample)

        for channel in range(62):
            trial_signal = tmp_trial_signal[channel]
            # 因为SEED数据没有基线信号部分

            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

            DE_delta = np.zeros(shape=[0], dtype=float)
            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(num_sample):
                DE_delta = np.append(DE_delta, compute_DE(delta[index * 100:(index + 1) * 100]))
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 100:(index + 1) * 100]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 100:(index + 1) * 100]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 100:(index + 1) * 100]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 100:(index + 1) * 100]))
            temp_de = np.vstack([temp_de, DE_delta])
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

        temp_trial_de = temp_de.reshape(-1, 5, num_sample)
        temp_trial_de = temp_trial_de.transpose([2, 0, 1])
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

    print("trial_DE shape:", decomposed_de.shape)
    return decomposed_de, label


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# 究极整合版
import os
import numpy as np

file_path = '/mnt/nvme0n1p1/EEG/SJTU/Preprocessed_EEG/'

people_name = ['dujingcheng_20131027', 'dujingcheng_20131030', 'dujingcheng_20131107',
               'mahaiwei_20130712', 'mahaiwei_20131016', 'mahaiwei_20131113',
               'penghuiling_20131027', 'penghuiling_20131030', 'penghuiling_20131106',
               'zhujiayi_20130709', 'zhujiayi_20131016', 'zhujiayi_20131105',
               'wuyangwei_20131127', 'wuyangwei_20131201', 'wuyangwei_20131207',
               'weiwei_20131130', 'weiwei_20131204', 'weiwei_20131211',
               'jianglin_20140404', 'jianglin_20140413', 'jianglin_20140419',
               'liuye_20140411', 'liuye_20140418', 'liuye_20140506',
               'sunxiangyu_20140511', 'sunxiangyu_20140514', 'sunxiangyu_20140521',
               'xiayulu_20140527', 'xiayulu_20140603', 'xiayulu_20140610',
               'jingjing_20140603', 'jingjing_20140611', 'jingjing_20140629',
               'yansheng_20140601', 'yansheng_20140615', 'yansheng_20140627',
               'wusifan_20140618', 'wusifan_20140625', 'wusifan_20140630',
               'wangkui_20140620', 'wangkui_20140627', 'wangkui_20140704',
               'liuqiujun_20140621', 'liuqiujun_20140702', 'liuqiujun_20140705']

short_name = ['djc', 'djc', 'djc', 'mhw', 'mhw', 'mhw', 'phl', 'phl', 'phl',
              'zjy', 'zjy', 'zjy', 'wyw', 'wyw', 'wyw', 'ww', 'ww', 'ww',
              'jl', 'jl', 'jl', 'ly', 'ly', 'ly', 'sxy', 'sxy', 'sxy',
              'xyl', 'xyl', 'xyl', 'jj', 'jj', 'jj', 'ys', 'ys', 'ys',
              'wsf', 'wsf', 'wsf', 'wk', 'wk', 'wk', 'lqj', 'lqj', 'lqj']

X = np.empty([0, 62, 5])
y = np.empty([0, 1])

for i in range(len(people_name)):
    file_name = file_path + people_name[i]
    print('processing {}'.format(people_name[i]))
    decomposed_de, label = decompose(file_name, short_name[i])
    X = np.vstack([X, decomposed_de])
    y = np.append(y, label)

np.save("/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/X_1D.npy", X)
np.save("/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/y.npy", y)




X = np.load('/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/X_1D.npy')
y = np.load('/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/y.npy')
# 生成8*9的矩阵形式
X89 = np.zeros((len(y), 8, 9, 5))
X89[:, 0, 2, :] = X[:, 3, :]
X89[:, 0, 3:6, :] = X[:, 0:3, :]
X89[:, 0, 6, :] = X[:, 4, :]
for i in range(5):
    X89[:, i + 1, :, :] = X[:, 5 + i * 9:5 + (i + 1) * 9, :]
X89[:, 6, 1:8, :] = X[:, 50:57, :]
X89[:, 7, 2:7, :] = X[:, 57:62, :]
np.save("/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/X89.npy", X89)
