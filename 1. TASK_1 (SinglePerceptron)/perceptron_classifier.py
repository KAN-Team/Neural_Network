import numpy as np
from sklearn.metrics import accuracy_score


class PerceptronClassifier:

    def signum_activation(self, vk):
        out = 1
        if vk < 0:
            out = -1
        return out

    def train_perceptron(self, feature1_train, feature2_train, Y_train, learning_rate, training_epochs, bias):
        neuron1_weights = [2.1, -1.1]

        for epoch in range(training_epochs):
            for i in range(len(feature1_train)):
                vk = (neuron1_weights[0] * feature1_train[i]) + (neuron1_weights[1] * feature2_train[i]) + bias
                y_predict = self.signum_activation(vk)
                if y_predict != Y_train[i]:  # Update the Weights
                    error = Y_train[i] - y_predict
                    neuron1_weights[0] = neuron1_weights[0] + (learning_rate * error * feature1_train[i])
                    neuron1_weights[1] = neuron1_weights[1] + (learning_rate * error * feature2_train[i])

        return neuron1_weights

    def classify_test(self, feature1_test, feature2_test, Y_test, weights, bias):
        y_prediction = []
        vk = np.dot(feature1_test, weights[0]) + np.dot(feature2_test, weights[1]) + bias
        for i in range(len(vk)):
            y_prediction.append(self.signum_activation(vk[i]))

        total_accuracy = accuracy_score(Y_test, y_prediction) * 100
        return total_accuracy, y_prediction
