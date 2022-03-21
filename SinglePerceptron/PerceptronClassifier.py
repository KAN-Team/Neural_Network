import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def concatinate_two_lists(first_list, second_list):
    out = [y for x in [first_list, second_list] for y in x]
    return out


# Remember Random (Shuffle)
def data_mapping(df):
    # Select Features (X1, X4) and Classes (1- Setosa, 2- versicolor)
    X1 = df['X1']
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

    X1_train = concatinate_two_lists(setosa_x1, versicolor_x1)
    X4_train = concatinate_two_lists(setosa_x4, versicolor_x4)
    X1_test = concatinate_two_lists(setosa_x1_test, versicolor_x1_test)
    X4_test = concatinate_two_lists(setosa_x4_test, versicolor_x4_test)

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


def signum_activation(vk):
    out = 1
    if vk < 0:
        out = -1
    return out


def train_perceptron(X1_train, X4_train, Y_train):
    learning_rate = 0.01
    training_epochs = 25
    bias = 1
    neuron1_weights = [0.2, 0.7]

    for epoch in range(training_epochs):
        for i in range(len(X1_train)):
            neuron1_net = float(neuron1_weights[0]) * float(X1_train[i])
            vk = neuron1_net + float((neuron1_weights[1]) * float(X4_train[i])) + bias
            y = signum_activation(vk)
            if y != Y_train[i]:  # update weights
                error = float(Y_train[i] - y)
                neuron1_weights[0] = float(neuron1_weights[0]) + (learning_rate * error * float(X1_train[i]))
                neuron1_weights[1] = float(neuron1_weights[1]) + (learning_rate * error * float(X4_train[i]))

    return neuron1_weights


if __name__ == "__main__":
    data_frame = read_dataset()
    dis_features_figures(data_frame)
    X1_train, X4_train, X1_test, X4_test, Y_train, Y_test = data_mapping(data_frame)
    weights = train_perceptron(X1_train, X4_train, Y_train)
