from TkinterGUI import TkinterGUI
import pandas as pd

if __name__ == '__main__':
    # read the given dataset...
    data = pd.read_csv('IrisData.txt')

    # Initialize GUI...
    frame = TkinterGUI(data=data)
    # print(frame.get_input_values())

    # from MLP import BackPropagation
    # backProp = BackPropagation(data=data, input_parameters=frame.get_input_values())
    # backProp.make_predictions_visualizations()  # this line is implicitly ran in BackPropagation initializer
