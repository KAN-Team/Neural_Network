from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox
from MLP import BackPropagation


# noinspection PyMethodMayBeStatic
class TkinterGUI:

    def __init__(self, data):
        # initialize Tkinter GUI fields response...
        self.data_arg = data
        self.bias_cb_r1 = 0
        self.actv_func_r2 = 0
        self.frame = 0
        self.entry_hidden_layers = 0
        self.entry_neurons_number = 0
        self.entry_learning_rate = 0
        self.entry_epochs_number = 0
        self.entry_hidden_layers_text = 0
        self.entry_neurons_number_text = 0
        self.entry_learning_rate_text = 0
        self.entry_epochs_number_text = 0

        # initialize widgets...
        self.__run_gui()

    def __run_gui(self):
        # Start Top Line
        self.frame = Tk()
        self.frame.geometry('600x400')
        self.frame.title('Back-Propagation Learning Algorithm')

        # TEXT VARIABLES
        self.bias_cb_r1 = IntVar()
        self.actv_func_r2 = StringVar()
        self.entry_hidden_layers_text = StringVar()
        self.entry_neurons_number_text = StringVar()
        self.entry_learning_rate_text = StringVar()
        self.entry_epochs_number_text = StringVar()

        # fill tkinter frame body with needed widgets
        self.__place_widgets()

        # initialize fields
        self.__initialize_widgets()

        self.frame.mainloop()
        # End Bottom Line

    def __place_widgets(self):
        # Place Header Label..
        Label(self.frame, text='Task3: Back-Propagation Learning Algorithm', font=26).place(x=130, y=20)

        # Place Labels...
        Label(self.frame, text='Enter Number of Hidden Layers: ').place(x=100, y=80)
        Label(self.frame, text='Enter Number of Neurons in each Layer: ').place(x=100, y=120)
        Label(self.frame, text='[N.B] Leave Spaces\nbetween layers\'\nneurons #', fg='#f00').place(x=480, y=110)
        Label(self.frame, text='Enter Learning rate (eta): ').place(x=100, y=160)
        Label(self.frame, text='Enter Number of epochs (m): ').place(x=100, y=200)
        Label(self.frame, text='Choose the Activation Function Type: ').place(x=100, y=240)

        # Place Entries...
        self.entry_hidden_layers = Entry(self.frame, textvariable=self.entry_hidden_layers_text).place(x=350, y=80)
        self.entry_neurons_number = Entry(self.frame, textvariable=self.entry_neurons_number_text).place(x=350, y=120)
        self.entry_learning_rate = Entry(self.frame, textvariable=self.entry_learning_rate_text).place(x=350, y=160)
        self.entry_epochs_number = Entry(self.frame, textvariable=self.entry_epochs_number_text).place(x=350, y=200)

        # Place Check box and Radio button...
        # bias check box
        bias_cb = Checkbutton(self.frame, text="Add Bias", variable=self.bias_cb_r1)
        bias_cb.place(x=280, y=280)

        # activation function radio button
        actv_func_cb = Combobox(self.frame, values=('Hyperbolic Tangent', 'Sigmoid'), width='17',
                                textvariable=self.actv_func_r2)
        actv_func_cb.place(x=350, y=240)

        # run button
        run_button = Button(self.frame, width='40', text='Run Back-Propagation Learning Algorithm',
                            command=self.run_learning_algorithm)
        run_button.place(x=160, y=320)

    def __initialize_widgets(self):
        self.entry_hidden_layers_text.set(2)
        self.entry_neurons_number_text.set('2 3')
        self.entry_learning_rate_text.set(0.0001)
        self.entry_epochs_number_text.set(36)
        self.actv_func_r2.set('Sigmoid')

    def __handle_invalid_fields(self):
        if self.entry_hidden_layers_text.get() == "":
            self.entry_hidden_layers_text.set(2)

        if self.entry_neurons_number_text.get() == "":
            self.entry_neurons_number_text.set('2, 3')

        if self.entry_learning_rate_text.get() == "":
            self.entry_learning_rate_text.set(0.0001)

        if self.entry_epochs_number_text.get() == "":
            self.entry_epochs_number_text.set(36)

        if self.actv_func_r2.get() != 'Sigmoid' and self.actv_func_r2.get() != 'Hyperbolic Tangent':
            self.actv_func_r2.set('Sigmoid')

    def get_input_values(self):
        return self.entry_hidden_layers_text.get(), self.entry_neurons_number_text.get(),\
               self.entry_learning_rate_text.get(), self.entry_epochs_number_text.get(), \
               self.actv_func_r2.get(), self.bias_cb_r1.get()

    def run_learning_algorithm(self):
        self.__handle_invalid_fields()
        BackPropagation(data=self.data_arg, input_parameters=self.get_input_values())
        # messagebox.showinfo('Task3: Back-Propagation Learning Algorithm', 'Learning Algorithm Finished Successfully!')
