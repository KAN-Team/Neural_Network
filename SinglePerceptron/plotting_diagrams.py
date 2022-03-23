import matplotlib.pyplot as plt
import numpy as np


class PlottingDiagrams:

    def draw_features_figures(self, df):
        for i in range(1, 5):
            for j in range(i + 1, 5):
                plt.figure('figure X{}, X{}'.format(i, j))
                plt.scatter(df['X{}'.format(i)][:50], df['X{}'.format(j)][:50])
                plt.scatter(df['X{}'.format(i)][50:100], df['X{}'.format(j)][50:100])
                plt.scatter(df['X{}'.format(i)][100:], df['X{}'.format(j)][100:])
                plt.xlabel('X{}'.format(i))
                plt.ylabel('X{}'.format(j))
                plt.show()

    def draw_learned_classes(self, type, num, feature1_train, feature2_train, weights):
        # w1x1 + w2x2 +bias = 0
        # x2 = ( (-w1x1) - bias ) / w2
        x2 = (-np.dot(feature1_train, weights[0])) / weights[1]

        plt.figure('figure of Perceptron {}'.format(type))
        plt.plot(feature1_train, x2, '-r', label='y= x1*W1 + X2*W2 + bias')
        plt.scatter(feature1_train[:num], feature2_train[:num])
        plt.scatter(feature1_train[num:], feature2_train[num:])
        plt.title('Graph of Perceptron {}'.format(type))
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper right')
        plt.show()
