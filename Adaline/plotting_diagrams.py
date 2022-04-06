import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class PlottingDiagrams:

    def draw_features_figures(self, df):
        for i in range(1, 5):
            for j in range(i + 1, 5):
                plt.figure('Features X{}, X{}'.format(i, j))
                plt.title('Features X{}, X{}'.format(i, j))
                plt.scatter(df['X{}'.format(i)][:50], df['X{}'.format(j)][:50])
                plt.scatter(df['X{}'.format(i)][50:100], df['X{}'.format(j)][50:100])
                plt.scatter(df['X{}'.format(i)][100:], df['X{}'.format(j)][100:])
                plt.xlabel('X{}'.format(i))
                plt.ylabel('X{}'.format(j))
                plt.show()

    def draw_learned_classes(self, type, num, feature1_train, feature2_train, weights, bias):
        # w0*bias + w1*x1 + w2*x2 = 0
        # x2 = ( (-w1*x1) - w0*bias ) / w2
        x2 = (-np.dot(feature1_train, weights[1]) - (weights[0] * bias)) / weights[2]

        plt.figure('figure of Adaline {}'.format(type))
        plt.plot(feature1_train, x2, '-r', label='bias*w0 + x1*W1 + X2*W2 = 0')
        plt.scatter(feature1_train[:num], feature2_train[:num])
        plt.scatter(feature1_train[num:], feature2_train[num:])
        plt.title('Graph of Adaline {}'.format(type))
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper right')
        plt.show()

    def draw_confusion_matrix(self, y_test, y_prediction):
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_prediction)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
