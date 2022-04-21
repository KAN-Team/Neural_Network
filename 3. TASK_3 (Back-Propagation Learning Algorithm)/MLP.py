import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# noinspection PyMethodMayBeStatic
class BackPropagation:
    # gets the samples data as the first argument
    # gets the input parameters from the GUI
    def __init__(self, data, input_parameters):
        train_data, test_data, input_parameters = self.__prepare_data(data, input_parameters)
        weights = self.__apply_algorithm(x_train=train_data[:, :-1], y_train=train_data[:, -1],
                                         input_parameters=input_parameters)
        self.n_actual, self.n_predict = self.__make_predictions(x_test=test_data[:, :-1], y_test=test_data[:, -1],
                                                                weights=weights, input_parameters=input_parameters)
        self.make_predictions_visualizations()

    # ===================================================
    def __prepare_data(self, data, input_parameters):
        # extracting features and labels from data
        x1, x2, x3, x4, label = data['X1'], data['X2'], data['X3'], data['X4'], data['Class']

        # providing one hot encoding for predictions simplicity
        samples_size = len(data)
        data = []
        for sampleIdx in range(samples_size):
            sample = [x1[sampleIdx], x2[sampleIdx], x3[sampleIdx], x4[sampleIdx]]
            if label[sampleIdx] == 'Iris-setosa':
                sample.append([1, 0, 0])
            elif label[sampleIdx] == 'Iris-versicolor':
                sample.append([0, 1, 0])
            else:
                sample.append([0, 0, 1])
            data.append(sample)

        print("Neurons List: ")
        print(input_parameters[1])
        print("================================\n")

        # splitting data into train and test portions
        train_data, test_data = self.__custom_train_test_split(data)

        if input_parameters[5] == 1:                            # if bias is enabled
            train_data = np.c_[np.ones(90, ), train_data]       # Adding ones column to the leftmost features
            test_data = np.c_[np.ones(60, ), test_data]

        # convert tuple to mutable list
        temp_list = list(input_parameters)
        temp_list[1] = list(temp_list[1].split(' '))
        input_parameters = tuple(temp_list)

        print("Train Data Head After Bias Checking: ")
        print(train_data[0:5, ])
        print("================================\n")

        return train_data, test_data, input_parameters

    def __custom_train_test_split(self, data):
        np.random.shuffle(data)
        train_data = []
        test_data = []
        setosaCount = 0
        versicolorCount = 0
        virginicaCount = 0
        for i in range(len(data)):
            if setosaCount < 30 and data[i][-1] == [1, 0, 0]:
                train_data.append([data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]])
                setosaCount = setosaCount + 1

            elif versicolorCount < 30 and data[i][-1] == [0, 1, 0]:
                train_data.append([data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]])
                versicolorCount = versicolorCount + 1

            elif virginicaCount < 30 and data[i][-1] == [0, 0, 1]:
                train_data.append([data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]])
                virginicaCount = virginicaCount + 1

            else:
                test_data.append([data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]])

        train_data = np.array(train_data)   # Convert to numpy array
        test_data = np.array(test_data)
        train_data.reshape(90, 5)           # Reshaping numpy array
        test_data.reshape(60, 5)

        print("After custom train test split")
        print("train data shape: ", train_data.shape)
        print("test data shape: ", test_data.shape)
        print("================================\n")

        return train_data, test_data
    # ===================================================

    def __apply_algorithm(self, x_train, y_train, input_parameters):
        weights = self.__get_random_weights(num_features=4, num_classes=3, input_parameters=input_parameters)

        for ep in range(int(input_parameters[3])):    # number of epochs
            for i in range(len(x_train)):
                outputs = self.__go_forward(x_sample=x_train[i], weights=weights, input_parameters=input_parameters)
                weights = self.__go_backward(x_sample=x_train[i], weights=weights,
                                             y_sample=y_train[i], outputs=outputs, input_parameters=input_parameters)

        return weights

    def __get_random_weights(self, num_features, num_classes, input_parameters):
        weights = []
        w = num_features   # initialized
        for L in range(int(input_parameters[0])):           # Number of Hidden Layers
            rand_mat = np.random.uniform(size=(int(input_parameters[1][L]), w + input_parameters[5]), low=-1, high=1)
            w = int(input_parameters[1][L])
            weights.append(rand_mat)

        # Output Layer
        weights.append(np.random.rand(num_classes, w + input_parameters[5]))
        return weights

    def __go_forward(self, x_sample, weights, input_parameters):
        outputs = []
        w_sz = len(weights)
        for i in range(w_sz):
            net_input = np.dot(weights[i], x_sample)
            c = [1] if input_parameters[5] == 1 and (i != w_sz - 1) else []

            for v in net_input:
                if input_parameters[4] == 'Sigmoid':
                    c.append(1 / (1 + np.exp(-v)))
                else:
                    c.append((1 - np.exp(-v)) / (1 + np.exp(-v)))

            outputs.append(c)
            x_sample = c

        return outputs

    def __go_backward(self, x_sample, weights, y_sample, outputs, input_parameters):
        C = []
        maxC = np.max(outputs[len(outputs) - 1])
        prevOutputs = outputs[len(outputs) - 1]

        for i in range(len(prevOutputs)):
            C.append(1 if prevOutputs[i] == maxC else 0)

        check = np.subtract(y_sample, C) == np.zeros(shape=(3, 1))
        if not check.all():
            error = []
            reversedOutputs = outputs.copy()
            reversedWeights = weights.copy()
            reversedOutputs.reverse()
            reversedWeights.reverse()

            for i in range(len(y_sample)):
                v = y_sample[i]
                c = reversedOutputs[0][i]
                error.append(((v - c) * c * (1 - c)) if input_parameters[4] == 'Sigmoid'
                             else ((v - c) * (1 - c) * (1 + c)))

            reversedOutputs.append(x_sample)
            for o in range(len(reversedOutputs) - 1):
                if o == 0:
                    for i in range(len(reversedOutputs[o+1])):
                        for j in range(len(reversedOutputs[o])):
                            reversedWeights[o][j, i] = reversedWeights[o][j, i] + \
                                                       float(input_parameters[2]) * error[j] * reversedOutputs[o+1][i]

                else:
                    av = 0 if input_parameters[4] == 'Sigmoid' else 1
                    for i in range(len(reversedOutputs[o+1])):
                        for j in range(1, len(reversedOutputs[o])):
                            soka = (av + reversedOutputs[o][j]) * (1 - reversedOutputs[o][j]) \
                                   * error[j-1] * reversedOutputs[o+1][i]
                            reversedWeights[o][j-1, i] = reversedWeights[o][j-1, i] + float(input_parameters[2]) * soka

                error = np.dot(error, reversedWeights[o][:, 1:]) if input_parameters[5] == 1 \
                    else np.dot(error, reversedWeights[o])

            for i in range(len(reversedWeights)):
                weights[len(reversedWeights) - 1 - i] = reversedWeights[i]

        return weights
    # ===================================================

    def __make_predictions(self, x_test, y_test, weights, input_parameters):
        y_pred = []  # get predicted forward y
        for i in range(len(x_test)):
            y_pred.append(self.__go_forward(x_sample=x_test[i], weights=weights, input_parameters=input_parameters)[-1])

        y_pred = np.array(y_pred)
        for i in range(len(y_pred)):
            maxC = np.max(y_pred[i])
            for j in range(len(y_pred[i])):
                y_pred[i][j] = 1 if y_pred[i][j] == maxC else 0


        print("Input Arguments:\n", input_parameters)
        n_predict = []
        n_actual = []
        for i in range(len(y_test)):
            res_1 = np.subtract(y_pred[i], [1, 0, 0]) == np.zeros(shape=(3, 1))
            res_2 = np.subtract(y_pred[i], [0, 1, 0]) == np.zeros(shape=(3, 1))
            if res_1.all():
                n_predict.append(0)
            elif res_2.all():
                n_predict.append(1)
            else:
                n_predict.append(2)

            res_1 = np.subtract(y_test[i], [1, 0, 0]) == np.zeros(shape=(3, 1))
            res_2 = np.subtract(y_test[i], [0, 1, 0]) == np.zeros(shape=(3, 1))
            if res_1.all():
                n_actual.append(0)
            elif res_2.all():
                n_actual.append(1)
            else:
                n_actual.append(2)

        return n_actual, n_predict

    def make_predictions_visualizations(self):
        cm = confusion_matrix(self.n_actual, self.n_predict)
        acc = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / 60 * 100
        print("TEST PREDICTIONS ACCURACY: %5.2f" % acc, "%")
        plt.figure(figsize=(8, 4))
        ax = sn.heatmap(cm, annot=True, cmap='Blues')
        plt.xlabel('Predict')
        plt.ylabel('Actual')
        ax.xaxis.set_ticklabels(['Setosa', 'Versicolor', 'Virginica'])
        ax.yaxis.set_ticklabels(['Setosa', 'Versicolor', 'Virginica'])
        plt.show()
    # ===================================================
