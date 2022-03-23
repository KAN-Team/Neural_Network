from tkinter import *
from dataset import Dataset
from perceptron_classifier import PerceptronClassifier
from plotting_diagrams import PlottingDiagrams


class UserInterface:
    root = Tk()

    def __init__(self):
        # Initialization
        self.dataset = Dataset()
        self.plotting = PlottingDiagrams()
        self.classifier = PerceptronClassifier()
        self.select_features = StringVar(self.root, "3")
        self.select_classes = StringVar(self.root, "1")
        self.learning_rate_var = DoubleVar(self.root, 0.0001)
        self.number_of_epochs_var = IntVar(self.root, 200)
        self.bias_checkbox = IntVar(self.root, 0)
        self.selected_feature1 = 'X1'
        self.selected_feature2 = 'X4'
        self.selected_class1 = 'Iris-setosa'
        self.selected_class2 = 'Iris-versicolor'

    def determine_features(self):
        if self.select_features.get() == '1':
            self.selected_feature1 = 'X1'
            self.selected_feature2 = 'X2'
        elif self.select_features.get() == '2':
            self.selected_feature1 = 'X1'
            self.selected_feature2 = 'X3'
        elif self.select_features.get() == '3':
            self.selected_feature1 = 'X1'
            self.selected_feature2 = 'X4'
        elif self.select_features.get() == '4':
            self.selected_feature1 = 'X2'
            self.selected_feature2 = 'X3'
        elif self.select_features.get() == '5':
            self.selected_feature1 = 'X2'
            self.selected_feature2 = 'X4'
        elif self.select_features.get() == '6':
            self.selected_feature1 = 'X3'
            self.selected_feature2 = 'X4'

    def determine_classes(self):
        if self.select_classes.get() == '1':
            self.selected_class1 = 'Iris-setosa'
            self.selected_class2 = 'Iris-versicolor'
        elif self.select_classes.get() == '2':
            self.selected_class1 = 'Iris-setosa'
            self.selected_class2 = 'Iris-virginica'
        elif self.select_classes.get() == '3':
            self.selected_class1 = 'Iris-versicolor'
            self.selected_class2 = 'Iris-virginica'

    def click_classify(self):
        self.determine_features()
        self.determine_classes()

        dataframe = self.dataset.read_dataset()
        self.plotting.draw_features_figures(dataframe)

        feature1_train, feature2_train, feature1_test, feature2_test, Y_train, Y_test = self.dataset.map_data(dataframe,
                         self.selected_feature1, self.selected_feature2, self.selected_class1, self.selected_class2)

        weights = self.classifier.train_perceptron(feature1_train, feature2_train, Y_train,
                  self.learning_rate_var.get(), self.number_of_epochs_var.get(), self.bias_checkbox.get())

        self.plotting.draw_learned_classes('Training', 30, feature1_train, feature2_train, weights)
        self.plotting.draw_learned_classes('Testing', 20, feature1_test, feature2_test, weights)

        self.classifier.classify_test(feature1_test, feature2_test, Y_test, weights)
        self.root.quit()

    def create_ui(self):
        self.root.title('Perceptron Classification App')
        self.root.geometry('840x600')

        welcome_title = StringVar()
        l = Label(self.root, textvariable=welcome_title)
        welcome_title.set("Welcome to Perceptron App")
        l.pack()

        Label(self.root, text="Select Features").place(x=40, y=40)
        Radiobutton(self.root, text="X1 & X2", value=1, variable=self.select_features,
                    indicator=1, background="light blue").place(x=40, y=80)
        Radiobutton(self.root, text="X1 & X3", value=2, variable=self.select_features,
                    indicator=1, background="light blue").place(x=130, y=80)
        Radiobutton(self.root, text="X1 & X4", value=3, variable=self.select_features,
                    indicator=1, background="light blue").place(x=220, y=80)
        Radiobutton(self.root, text="X2 & X3", value=4, variable=self.select_features,
                    indicator=1, background="light blue").place(x=310, y=80)
        Radiobutton(self.root, text="X2 & X4", value=5, variable=self.select_features,
                    indicator=1, background="light blue").place(x=400, y=80)
        Radiobutton(self.root, text="X3 & X4", value=6, variable=self.select_features,
                    indicator=1, background="light blue").place(x=490, y=80)

        Label(self.root, text="Select Classes").place(x=40, y=130)
        Radiobutton(self.root, text="C1 & C2", value=1, variable=self.select_classes,
                    indicator=2, background="light blue").place(x=40, y=170)
        Radiobutton(self.root, text="C1 & X3", value=2, variable=self.select_classes,
                    indicator=2, background="light blue").place(x=130, y=170)
        Radiobutton(self.root, text="C2 & C3", value=3, variable=self.select_classes,
                    indicator=2, background="light blue").place(x=220, y=170)

        Label(self.root, text="Learning Rate").place(x=40, y=200)
        Entry(self.root, textvariable=self.learning_rate_var, width=30).place(x=180, y=200)

        Label(self.root, text="Number of Epochs").place(x=40, y=250)
        Entry(self.root, textvariable=self.number_of_epochs_var, width=30).place(x=180, y=250)

        Checkbutton(self.root, text="Add Bias", variable=self.bias_checkbox).place(x=40, y=300)
        Button(self.root, text="Classify", width=15, command=self.click_classify).place(x=40, y=350)
        self.root.mainloop()


if __name__ == "__main__":
    ui = UserInterface()
    ui.create_ui()
