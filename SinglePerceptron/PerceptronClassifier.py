import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
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
    out = []
    for i in range(len(first_list)):
        out.append(float(first_list[i]))
    for j in range(len(second_list)):
        out.append(float(second_list[j]))
    return out


# Remember Random (Shuffle)
def map_data(df):
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


def train_perceptron(feature1_train, feature2_train, Y_train):
    learning_rate = 0.0001
    training_epochs = 200
    bias = 1
    neuron1_weights = [2.1, -1.1]

    for epoch in range(training_epochs):
        for i in range(len(feature1_train)):
            vk = (neuron1_weights[0] * feature1_train[i]) + (neuron1_weights[1] * feature2_train[i]) + bias
            y_predict = signum_activation(vk)
            if y_predict != Y_train[i]:  # Update Weights
                error = Y_train[i] - y_predict
                neuron1_weights[0] = neuron1_weights[0] + (learning_rate * error * feature1_train[i])
                neuron1_weights[1] = neuron1_weights[1] + (learning_rate * error * feature2_train[i])

    return neuron1_weights


def draw_learned_classes(feature1_train, feature2_train,  Y_train, weights):
    y = np.dot(feature1_train, weights[0])

    plt.figure('figure of Perceptron')
    plt.plot(feature1_train, y, '-r', label='y= x1*W1 + X2*W2 + bias')
    plt.scatter(feature1_train[:30], Y_train[:30])
    plt.scatter(feature1_train[30:], Y_train[30:])
    plt.title('Graph of Perceptron')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper right')
    plt.show()


def classify_test(feature1_test, feature2_test, Y_test, weights):
    y_prediction = []
    vk = np.dot(feature1_test, weights[0]) + np.dot(feature2_test, weights[1])
    for i in range(len(vk)):
        y_prediction.append(signum_activation(vk[i]))

    print("Total Accuracy: {}%".format(accuracy_score(Y_test, y_prediction) * 100))


if __name__ == "__main__":
    data_frame = read_dataset()
    # dis_features_figures(data_frame)
    feature1_train, feature2_train, feature1_test, feature2_test, Y_train, Y_test = map_data(data_frame)
    weights = train_perceptron(feature1_train, feature2_train, Y_train)
    # draw_learned_classes(feature1_train, feature2_train,  Y_train, weights)
    classify_test(feature1_test, feature2_test, Y_test, weights)
