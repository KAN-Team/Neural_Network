import numpy as np
import random
import math
from sklearn.metrics import accuracy_score


class AdalineClassifier:
    def __init__(self):
        self.total_error = 0

    def signum_activation(self, vk):
        out = 1
        if vk < 0:
            out = -1
        return out


    def train_adaline(self, feature1_train, feature2_train, Y_train, learning_rate, training_epochs, threshold, bias):
        neuron1_weights = [random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3)]

        for epoch in range(training_epochs):
            for i in range(len(feature1_train)):
                vk = (neuron1_weights[0] * bias) + (neuron1_weights[1] * feature1_train[i]) + (neuron1_weights[2]
                                                                                               * feature2_train[i])

                y_predict = vk
                error = Y_train[i] - y_predict
                # update the weights
                neuron1_weights[0] = neuron1_weights[0] + (learning_rate * error * bias)
                neuron1_weights[1] = neuron1_weights[1] + (learning_rate * error * feature1_train[i])
                neuron1_weights[2] = neuron1_weights[2] + (learning_rate * error * feature2_train[i])

            # loop for calculating MES
            for i in range(len(feature1_train)):
                new_y_predict = (neuron1_weights[0] * bias) + (neuron1_weights[1] *
                                                               feature1_train[i]) + (
                                            neuron1_weights[2] * feature2_train[i])

                self.total_error = self.total_error + (0.5 * (math.pow(Y_train[i] - new_y_predict, 2)))

            # condition for error bounds
            mse = (1 / len(feature1_train)) * self.total_error
            if mse < threshold:
                return neuron1_weights

        return neuron1_weights

    def classify_test(self, feature1_test, feature2_test, Y_test, weights, bias):
        y_prediction = []
        vk = (weights[0] * bias) + np.dot(feature1_test, weights[1]) + np.dot(feature2_test, weights[2])
        for i in range(len(vk)):
            y_prediction.append(self.signum_activation(vk[i]))

        total_accuracy = accuracy_score(Y_test, y_prediction) * 100
        return total_accuracy, y_prediction
