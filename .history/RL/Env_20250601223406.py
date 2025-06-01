# encoding=utf-8
import csv
from math import log

import numpy as np
import os
import gym
import sklearn
import torch
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from torch import nn


# classification_report：输出模型评估报告; accuracy_score：返回分类正确数量的百分比; roc_auc_score: 计算AUC值；


class ClassifyEnv(gym.Env):

    def __init__(self, mode, imb_rate, trainx, trainy, lamda=0.1):  # mode means training or testing
        self.mode = mode
        self.imb_rate = imb_rate
        self.lamda = lamda
        self.min = 0
        self.maj = 0

        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])

        self.game_len = self.Env_data.shape[0]

        self.num_classes = len(set(self.Answer))
        self.action_space = spaces.Discrete(self.num_classes)  # 创建一个离散的num_classes维空间
        print(self.action_space)
        self.step_ind = 0
        self.y_pred = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a, predict):
        self.y_pred.append(a)
        self.info = {}
        terminal = False
        # 自适应调整的不均衡率
        if self.Answer[self.id[self.step_ind]] == 1:
            self.min += 1
        else:
            self.maj += 1
        if a == self.Answer[self.id[self.step_ind]]:
            if self.Answer[self.id[self.step_ind]] == 1:
                reward = 1.
            else:
                reward = 1. * self.imb_rate
        else:
            if self.Answer[self.id[self.step_ind]] == 1:
                reward = -1. - self.lamda * (1. / (self.step_ind + 1))
                if self.mode == 'train':
                    terminal = True
            else:
                reward = -1. * self.imb_rate - self.lamda * (1. / (self.step_ind + 1))

        # 原始奖励函数
        # if a == self.Answer[self.id[self.step_ind]]:
        #     reward = 1.
        # else:
        #     reward = -1.
        #     if self.mode == 'train':
        #         terminal = True

        self.step_ind += 1

        if self.step_ind == self.game_len:
            if self.mode == 'test':
                y_true_cur = self.Answer[self.id]
                self.info['acc'], self.info['precision'], self.info['recall'], \
                    self.info['fmeasure'], self.info['auc'], self.info['cm'], self.info['predict'] = self.My_metrics(
                    np.array(self.y_pred),
                    np.array(y_true_cur[
                             :self.step_ind]), predict)
            terminal = True

        # 自适应调整的不均衡率
        if terminal is True and self.maj != 0:
            self.imb_rate = self.min / self.maj

        return self.Env_data[self.id[self.step_ind % self.game_len]], reward, terminal, self.info

    def My_metrics(self, y_pre, y_true, pre):
        # 读取片段列表
        # csv_file_path = 'data/daic/test.csv'
        # test_name = []
        # with open(csv_file_path, mode='r', newline='') as file:
        #     reader = csv.reader(file)
        #     for row in reader:
        #         test_name.extend(row)
        #
        # # 读取测试集列表
        # test_file_path = 'data/daic/dev_split_Depression_AVEC2017.csv'
        # samples = []
        # labels = []
        # with open(test_file_path, mode='r', newline='') as file:
        #     reader = csv.reader(file)
        #     next(reader)
        #     for row in reader:
        #         samples.append(row[0])
        #         labels.append(int(row[1]))

        confusion_mat = confusion_matrix(y_true, y_pre)
        print('\n')
        print(classification_report(y_true, y_pre))
        conM = np.array(confusion_mat, dtype='float')
        TP = conM[1][1]
        TN = conM[0][0]
        FN = conM[1][0]
        FP = conM[0][1]
        TPrate = TP / (TP + FN)  # 真阳性率
        TNrate = TN / (TN + FP)  # 真阴性率
        FPrate = FP / (TN + FP)  # 假阳性率
        FNrate = FN / (TP + FN)  # 假阴性率
        PPvalue = TP / (TP + FP)  # 阳性预测值
        NPvalue = TN / (TN + FN)  # 假性预测值

        G_mean = np.sqrt(TPrate * TNrate)

        Recall = TPrate = TP / (TP + FN)
        Precision = PPvalue = TP / (TP + FP)
        acc = (TP + TN) / (TP + TN + FN + FP)
        F_measure = 2 * Recall * Precision / (Recall + Precision)
        auc = roc_auc_score(y_true, y_pre)

        print(confusion_mat)
        res = 'Acc:{}, Precision:{}, Recall:{},F_measure:{},Auc:{}\n' \
            .format(acc, Precision, Recall, F_measure, auc)
        print(res)
        print()
        return acc, Precision, Recall, F_measure, auc, conM, y_pre

    # return: (states, observations)
    def reset(self):
        if self.mode == 'train':
            np.random.shuffle(self.id)
        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]

    def get_info(self):
        return self.info
