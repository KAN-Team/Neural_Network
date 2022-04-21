from tkinter import *
from dataset import Dataset
from adaline_classifier import AdalineClassifier
from plotting_diagrams import PlottingDiagrams

class UserInterface:
    root = Tk()

    def __init__(self):
        # Initialization
        self.dataset = Dataset()
        self.plotting = PlottingDiagrams()
        self.classifier = AdalineClassifier()
        self.select_features = StringVar(self.root, "3")
        self.select_classes = StringVar(self.root, "1")
        self.learning_rate_var = DoubleVar(self.root, 0.0001)
        self.number_of_epochs_var = IntVar(self.root, 200)
        self.mse_threshold = DoubleVar(self.root, 0.1)
        self.bias_checkbox = IntVar(self.root, 0)
        self.selected_feature1 = 'X1'
        self.selected_feature2 = 'X4'
        self.selected_class1 = 'Iris-setosa'
        self.selected_class2 = 'Iris-versicolor'
        self.total_accuracy = 0
        self.accuracy_label = StringVar()

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

        feature1_train, feature2_train, feature1_test, feature2_test, Y_train, Y_test = self.dataset.map_data(dataframe,
                         self.selected_feature1, self.selected_feature2, self.selected_class1, self.selected_class2)

        weights = self.classifier.train_adaline(feature1_train, feature2_train, Y_train,
                  self.learning_rate_var.get(), self.number_of_epochs_var.get(),
                  self.mse_threshold.get(), self.bias_checkbox.get())

        self.plotting.draw_learned_classes('Training', 30, feature1_train, feature2_train,
                                           weights, self.bias_checkbox.get())
        self.plotting.draw_learned_classes('Testing', 20, feature1_test, feature2_test,
                                           weights, self.bias_checkbox.get())

        self.total_accuracy, y_predict = self.classifier.classify_test(feature1_test, feature2_test, Y_test,
                                                                       weights, self.bias_checkbox.get())

        self.plotting.draw_confusion_matrix(Y_test, y_predict)
        self.accuracy_label.set("{}%".format(self.total_accuracy))
        #self.root.quit()


    def click_visualize(self):
        dataframe = self.dataset.read_dataset()
        self.plotting.draw_features_figures(dataframe)

    def create_ui(self):
        self.root.title('Adaline Classification App')
        self.root.geometry('840x600')
        self.root.config(bg="#DECBA4")

        welcome_title = StringVar()
        l = Label(self.root, textvariable=welcome_title)
        welcome_title.set("/Welcome to Adaline App\\")
        l.config(font=('Comic Sans MS bold', 15), fg="#000", bg="#DECBA4")
        l.pack()

        l = Label(self.root, text="Select Features:")
        l.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        l.place(x=40, y=40)

        Radiobutton(self.root, text="X1 & X2", value=1, variable=self.select_features,
                    indicator=1, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=40, y=80)
        Radiobutton(self.root, text="X1 & X3", value=2, variable=self.select_features,
                    indicator=1, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=130, y=80)
        Radiobutton(self.root, text="X1 & X4", value=3, variable=self.select_features,
                    indicator=1, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=220, y=80)
        Radiobutton(self.root, text="X2 & X3", value=4, variable=self.select_features,
                    indicator=1, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=310, y=80)
        Radiobutton(self.root, text="X2 & X4", value=5, variable=self.select_features,
                    indicator=1, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=400, y=80)
        Radiobutton(self.root, text="X3 & X4", value=6, variable=self.select_features,
                    indicator=1, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=490, y=80)

        l = Label(self.root, text="Select Classes:")
        l.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        l.place(x=40, y=130)

        Radiobutton(self.root, text="C1 & C2", value=1, variable=self.select_classes,
                    indicator=2, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=40, y=170)
        Radiobutton(self.root, text="C1 & X3", value=2, variable=self.select_classes,
                    indicator=2, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=130, y=170)
        Radiobutton(self.root, text="C2 & C3", value=3, variable=self.select_classes,
                    indicator=2, font=('Comic Sans MS bold', 10), fg="black", bg="#DECBA4"
                    ).place(x=220, y=170)

        l = Label(self.root, text="Learning Rate:")
        l.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        l.place(x=40, y=220)

        e = Entry(self.root, textvariable=self.learning_rate_var, width=30)
        e.config(font=('Comic Sans MS bold', 10), fg="#000", bg="#FFF")
        e.place(x=210, y=225)

        l = Label(self.root, text="Number of Epochs:")
        l.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        l.place(x=40, y=265)

        e = Entry(self.root, textvariable=self.number_of_epochs_var, width=30)
        e.config(font=('Comic Sans MS bold', 10), fg="#000", bg="#FFF")
        e.place(x=210, y=270)

        l = Label(self.root, text="MSE Threshold:")
        l.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        l.place(x=40, y=310)

        e = Entry(self.root, textvariable=self.mse_threshold, width=30)
        e.config(font=('Comic Sans MS bold', 10), fg="#000", bg="#FFF")
        e.place(x=210, y=315)

        l = Label(self.root, text="Overall Accuracy:")
        l.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        l.place(x=300, y=355)
        l = Label(self.root, textvariable=self.accuracy_label)
        l.config(font=('Comic Sans MS bold', 10), fg="#FF0000", bg="#DECBA4")
        l.place(x=450, y=358)

        c = Checkbutton(self.root, text="Add Bias", variable=self.bias_checkbox)
        c.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#DECBA4")
        c.place(x=40, y=350)
        b = Button(self.root, text="Classify", width=15, command=self.click_classify)
        b.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#FFF")
        b.place(x=100, y=420)

        b = Button(self.root, text="Visualize Plotting", width=15, command=self.click_visualize)
        b.config(font=('Comic Sans MS bold', 12), fg="#0000FF", bg="#FFF")
        b.place(x=280, y=420)

        self.root.mainloop()


if __name__ == "__main__":
    ui = UserInterface()
    ui.create_ui()
