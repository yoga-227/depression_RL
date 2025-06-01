import os
import pickle
import random
import time

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt

from RL.RL_train import one_train
from dataset import CMDC_AUDIO_DATASET
from split_datasets import generator_dataset_from_indexs_group_by_id, get_dataset_from_folder

root = '../data/CMDC'
save_path = '../save/'
num_classes = 2
BATCH_SIZE = 8
folds = 5
dataset_name = 'CMDC'

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(ds):
    x = []
    y = []
    for feature, label in ds:
        x.append(feature)
        y.append(label)
    return x, y


def plot_confusion_matrix(fold, confusion_matrix, labels):
    class_totals = np.sum(confusion_matrix, axis=1)
    percentage_matrix = confusion_matrix / class_totals[:, np.newaxis]
    percentage_matrix *= 100
    rounded_percentage_matrix = np.round(percentage_matrix, decimals=2)

    # 创建热力图
    ax = sns.heatmap(rounded_percentage_matrix, annot=True, fmt=".2f", cmap="Blues")

    # 设置轴标签
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # 设置类别标签
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # 显示热力图
    save_path = '../log/confusion_matrix_D3QN/{}'.format(dataset_name)
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/' + 'fold{}'.format(fold) + '.png'
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.show()
    plt.close()


Fold_1 = [np.arange(1, 21), np.arange(1, 41), np.arange(21, 27), np.arange(41, 53)]
Fold_2 = [np.arange(7, 27), np.arange(13, 53), np.arange(1, 7), np.arange(1, 13)]
Fold_3 = [np.append(np.arange(1, 7), np.arange(13, 27)), np.append(np.arange(1, 13), np.arange(25, 53)),
          np.arange(7, 13), np.arange(13, 25)]
Fold_4 = [np.append(np.arange(1, 13), np.arange(19, 27)), np.append(np.arange(1, 25), np.arange(37, 53)),
          np.arange(13, 19), np.arange(25, 37)]
Fold_5 = [np.append(np.arange(1, 19), np.arange(25, 27)), np.append(np.arange(1, 37), np.arange(49, 53)),
          np.arange(19, 25), np.arange(37, 49)]

Folds = [Fold_1, Fold_2, Fold_3, Fold_4, Fold_5]

ACC = {}
Precision = {}
Recall = {}
F1_score = {}
AUC = {}

for fold in range(5):
    start = time.time()
    generator = generator_dataset_from_indexs_group_by_id
    train_dataset, test_dataset = get_dataset_from_folder(Folds[fold], generator=generator, suffixs=['wav.csv'],
                                                          root=root)
    x_tag = [i.replace('.', '') for i in ['wav.csv']]
    train_dataset = CMDC_AUDIO_DATASET(train_dataset, x_tag)
    test_dataset = CMDC_AUDIO_DATASET(test_dataset, x_tag)

    x_train, y_train = load_data(train_dataset)
    x_test, y_test = load_data(test_dataset)

    x_train = np.squeeze(np.array(x_train))
    y_train = np.squeeze(np.array(y_train))
    x_test = np.squeeze(np.array(x_test))
    y_test = np.squeeze(np.array(y_test))

    # # 将 NumPy 数组写入 .pkl 文件
    # file_path = '../data/cmdc/fold{}.pkl'.format(fold)
    # with open(file_path, 'wb') as file:
    #     pickle.dump(x_train, file)
    #     pickle.dump(y_train, file)
    #     pickle.dump(x_test, file)
    #     pickle.dump(y_test, file)

    fold_acc, fold_prec, fold_recall, fold_f1, fold_auc, fold_cm = one_train(fold, x_train, y_train, x_test, y_test)

    # 绘制混淆矩阵
    labels = ['HC', 'MDD']
    plot_confusion_matrix(fold, fold_cm, labels)

    key = 'fold{}'.format(fold)
    ACC[key] = fold_acc
    Precision[key] = fold_prec
    Recall[key] = fold_recall
    F1_score[key] = fold_f1
    AUC[key] = fold_auc

    end = time.time()
    consume_time = end - start
    print('all time:', consume_time)

    # dicts = [Accuracy, Precision, Recall, F1, Auc]
    # res_other = join_multi_dicts(dicts)
    # write_excel_dict(res_other, dataset_name, 'im_0.2')
print("每一折的Accuracy:", ACC)
print("每一折的Precision:", Precision)
print("每一折的Recall:", Recall)
print("每一折的F1-score:", F1_score)
print("每一折的AUC:", AUC)
