import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def read_dataset():
    dataset_file = open("IrisData.txt", "r")
    cnt = 0
    all_samples = []

    for sample in dataset_file:
        if cnt == 0:
            cnt = cnt + 1
            continue
        else:
            splited_sample = sample.split(",")
            all_samples.append(splited_sample)
        cnt = cnt + 1

    df = pd.DataFrame(all_samples, columns=['X1', 'X2', 'X3', 'X4', 'Y'])
    # print(df['X1'])
    return df


def dis_features_figures(df):
    for i in range(1, 5):
        for j in range(i + 1, 5):
            plt.figure('figure X{}, X{}'.format(i, j))
            plt.scatter(df['X{}'.format(i)], df['Y'])
            plt.scatter(df['X{}'.format(j)], df['Y'])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()


def data_mapping(df):
    # Select Features (X1, X4) and Classes (1- Setosa, 2- versicolor)
    X1 = df['X1']
    X4 = df['X4']
    target = df['Y'][:100]
    Y = []
    # 30 sample for training
    setosa_x1 = np.array(X1[:30])
    setosa_x4 = np.array(X4[:30])
    versicolor_x1 = np.array(X1[50:80])
    versicolor_x4 = np.array(X4[50:80])

    # 20 sample for testing
    setosa_x1_test = np.array(X1[30:50])
    setosa_x4_test = np.array(X4[30:50])
    versicolor_x1_test = np.array(X1[80:100])
    versicolor_x4_test = np.array(X4[80:100])

    # Y Mapping into 0 for setosa class and 1 for virginica class
    for val in target:
        if val == 'Iris-setosa\n':
            Y.append(0)
        else:
            Y.append(1)

    # print(setosa_x1)


if __name__ == "__main__":
    data_frame = read_dataset()
    # dis_features_figures(data_frame)
    # X1 and X4 is discriminative
    data_mapping(data_frame)
