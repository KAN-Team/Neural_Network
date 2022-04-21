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

    def select_classes_data(self, class1_feature1, class1_feature2, class1_feature1_tst, class1_feature2_tst,
                            class2_feature1, class2_feature2, class2_feature1_tst, class2_feature2_tst):

        feature1_train = self.concatenate_two_lists(class1_feature1, class2_feature1)
        feature2_train = self.concatenate_two_lists(class1_feature2, class2_feature2)
        feature1_test = self.concatenate_two_lists(class1_feature1_tst, class2_feature1_tst)
        feature2_test = self.concatenate_two_lists(class1_feature2_tst, class2_feature2_tst)
        return feature1_train, feature2_train, feature1_test, feature2_test

    def map_label(self, class1, class2):
        lst1 = []
        lst2 = []
        for val in range(len(class1)):
            lst1.append(1)
        for val in range(len(class2)):
            lst1.append(-1)

        return lst1, lst2

    def map_data(self, df, feature1, feature2, cls1, cls2):
        class1 = df.iloc[:50, :]
        class2 = df.iloc[50:100, :]
        class3 = df.iloc[100:, :]

        # Random selection 30 sample for training and 20 sample for testing
        class1_train = class1.sample(frac=0.6)
        class1_test = class1.drop(pd.DataFrame(class1_train).index[:])
        class2_train = class2.sample(frac=0.6)
        class2_test = class2.drop(pd.DataFrame(class2_train).index[:])
        class3_train = class3.sample(frac=0.6)
        class3_test = class3.drop(pd.DataFrame(class3_train).index[:])

        # Training Data
        setosa_feature1 = np.array(class1_train[feature1])
        setosa_feature2 = np.array(class1_train[feature2])
        versicolor_feature1 = np.array(class2_train[feature1])
        versicolor_feature2 = np.array(class2_train[feature2])
        virginica_feature1 = np.array(class3_train[feature1])
        virginica_feature2 = np.array(class3_train[feature2])

        # Testing Data
        setosa_feature1_tst = np.array(class1_test[feature1])
        setosa_feature2_tst = np.array(class1_test[feature2])
        versicolor_feature1_tst = np.array(class2_test[feature1])
        versicolor_feature2_tst = np.array(class2_test[feature2])
        virginica_feature1_tst = np.array(class3_test[feature1])
        virginica_feature2_tst = np.array(class3_test[feature2])

        if cls1 == 'Iris-setosa' and cls2 == 'Iris-versicolor':
            feature1_train, feature2_train, feature1_test, feature2_test = self.select_classes_data(
                setosa_feature1, setosa_feature2, setosa_feature1_tst, setosa_feature2_tst,
                versicolor_feature1, versicolor_feature2, versicolor_feature1_tst, versicolor_feature2_tst)

            x1, x2 = self.map_label(class1_train['Y'], class2_train['Y'])
            Y_train = self.concatenate_two_lists(x1, x2)
            x1, x2 = self.map_label(class1_test['Y'], class2_test['Y'])
            Y_test = self.concatenate_two_lists(x1, x2)


        elif cls1 == 'Iris-setosa' and cls2 == 'Iris-virginica':
            feature1_train, feature2_train, feature1_test, feature2_test = self.select_classes_data(
                setosa_feature1, setosa_feature2, setosa_feature1_tst, setosa_feature2_tst,
                virginica_feature1, virginica_feature2, virginica_feature1_tst, virginica_feature2_tst)
            print("Kareem")
            x1, x2 = self.map_label(class1_train['Y'], class3_train['Y'])
            Y_train = self.concatenate_two_lists(x1, x2)
            x1, x2 = self.map_label(class1_test['Y'], class3_test['Y'])
            Y_test = self.concatenate_two_lists(x1, x2)


        else:
            feature1_train, feature2_train, feature1_test, feature2_test = self.select_classes_data(
                versicolor_feature1, versicolor_feature2, versicolor_feature1_tst, versicolor_feature2_tst,
                virginica_feature1, virginica_feature2, virginica_feature1_tst, virginica_feature2_tst)

            x1, x2 = self.map_label(class2_train['Y'], class3_train['Y'])
            Y_train = self.concatenate_two_lists(x1, x2)
            x1, x2 = self.map_label(class2_test['Y'], class3_test['Y'])
            Y_test = self.concatenate_two_lists(x1, x2)

        print("The Size of Feature 1 Training: ", len(feature1_train))
        print("The Size of Feature 2 Training: ", len(feature2_train))
        print("The Size of Y Training: ", len(Y_train))
        print("The Size of Feature 1 Testing: ", len(feature1_test))
        print("The Size of Feature 2 Testing: ", len(feature2_test))
        print("The Size of Y Testing: ", len(Y_test))

        return feature1_train, feature2_train, feature1_test, feature2_test, Y_train, Y_test
