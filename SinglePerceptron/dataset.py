import pandas as pd
import numpy as np


class Dataset:

    def __init__(self):
        pass

    def read_dataset(self):
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
        return df

    def concatenate_two_lists(self, first_list, second_list):
        out = []
        for i in range(len(first_list)):
            out.append(float(first_list[i]))
        for j in range(len(second_list)):
            out.append(float(second_list[j]))
        return out

    def map_data(self, df):
        X1 = df['X2']
        X4 = df['X4']
        target = df['Y'][:100]
        Y = []
        Y_train = []
        Y_test = []

        # Y Mapping into 0 for setosa class and 1 for virginica class
        for val in target:
            if val == 'Iris-setosa\n':
                Y.append(1)
            else:
                Y.append(-1)

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

        X1_train = self.concatenate_two_lists(setosa_x1, versicolor_x1)
        X4_train = self.concatenate_two_lists(setosa_x4, versicolor_x4)
        X1_test = self.concatenate_two_lists(setosa_x1_test, versicolor_x1_test)
        X4_test = self.concatenate_two_lists(setosa_x4_test, versicolor_x4_test)

        for i in range(len(Y)):
            if i >= 80:
                break
            elif i >= 30 and i < 50:
                continue
            else:
                Y_train.append(Y[i])

        for i in range(30, len(Y)):
            if i >= 50 and i < 80:
                continue
            else:
                Y_test.append(Y[i])

        return X1_train, X4_train, X1_test, X4_test, Y_train, Y_test

# Remember Random (Shuffle)
