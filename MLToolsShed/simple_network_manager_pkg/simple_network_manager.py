import inspect
import numpy as np
from keras.layers import Dense
from keras import Input, Model, optimizers
from MLToolsShed.utilities_pkg import runtime_error_handler


class SimpleNetworkManager:

    def __init__(self, number_of_classes, learning_rate, hidden_neurons_cards, activation_fncts, epochs, batch_size,
                 input_x_dimension):

        np.random.seed(1234)

        #   ml hyper-parameters
        self.input_x_dimension = input_x_dimension
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.hidden_neurons_cards = hidden_neurons_cards
        self.activation_fncts = activation_fncts
        self.epochs = epochs
        self.batch_size = batch_size

        #   ml store vectors
        self.classifier_network_epochs = []
        self.classifier_network_loss_vec = []
        self.classifier_network_categ_acc_vec = []
        self.classifier_network_val_loss_vec = []
        self.classifier_network_val_categ_acc_vec = []
        self.f1_value_training = []
        self.f1_value_validation = []

        self.results_folder = None

        self.check_activation_fncts_length()
        self.model = self.build_simple_network()

    def check_activation_fncts_length(self):
        if len(self.activation_fncts) != len(self.hidden_neurons_cards) + 1:
            runtime_error_handler.runtime_error_handler(str_='loss_mismatch', add=str(inspect.currentframe()))

    def build_simple_network(self):
        #   inputs of dimension input_x_dimension
        input_layer = Input(shape=(self.input_x_dimension,), name='classifier_network_input')

        if len(self.hidden_neurons_cards) > 0:
            #   at least one hidden layer
            hidden_layer = Dense(self.hidden_neurons_cards[0],
                                 activation=self.activation_fncts[0],
                                 name='classifier_network_hidden_layer_0')(input_layer)

            #   other hidden layers if any
            for idx in range(1, len(self.hidden_neurons_cards)):
                layer_name = 'classifier_network_hidden_layer_' + str(idx)
                hidden_layer = Dense(self.hidden_neurons_cards[idx],
                                     activation=self.activation_fncts[idx],
                                     name=layer_name)(hidden_layer)

            #   output layer: since it is a softmax layer a neuron is needed to encode each class
            output_layer = Dense(
                self.number_of_classes,
                activation='softmax',
                name='classifier_network_output_layer')(hidden_layer)

        else:
            #   output layer: since it is a softmax layer a neuron is needed to encode each class
            output_layer = Dense(
                self.number_of_classes,
                activation='softmax',
                name='classifier_network_output_layer')(input_layer)

        #   create the network model
        classifier_network_model = Model(inputs=input_layer,
                                         outputs=output_layer,
                                         name='classifier_network_model')

        if self.learning_rate == "None":
            classifier_network_model.compile(loss='categorical_crossentropy',
                                             optimizer='adam',
                                             metrics=['categorical_accuracy'])

        else:
            opt = optimizers.Adam(lr=float(self.learning_rate))
            classifier_network_model.compile(loss='categorical_crossentropy',
                                             optimizer=opt,
                                             metrics=['categorical_accuracy'])

        print(classifier_network_model.summary(line_length=100))

        return classifier_network_model
