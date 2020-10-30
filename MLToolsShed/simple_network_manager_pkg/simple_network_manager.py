import pickle
import inspect
import numpy as np
from keras.layers import Dense
from keras import Input, Model, optimizers
from MLToolsShed.utilities_pkg import gpu_setup, runtime_error_handler, utilities


class SimpleNetworkManager:

    def __init__(self, id_gpu, perc_gpu, learning_rate, hidden_neurons_cards, activation_fncts, epochs,
                 batch_size, results_folder, metrics_list, loss='categorical_crossentropy'):

        np.random.seed(1234)

        #   ml hyper-parameters
        self.input_x_dimension = None
        self.number_of_classes = None
        self.learning_rate = learning_rate
        self.hidden_neurons_cards = hidden_neurons_cards
        self.activation_fncts = activation_fncts
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.metrics = metrics_list
        self.id_gpu = id_gpu
        self.perc_gpu = perc_gpu

        #   ml store vectors
        self.classifier_network_loss_vec = []
        self.classifier_network_metrics_vec = []
        self.classifier_network_val_loss_vec = []
        self.classifier_network_val_metrics_vec = []
        self.classifier_network_evaluation_on_test_set_loss_vec = []
        self.classifier_network_evaluation_on_test_set_metrics_vec = []
        self.f1_value_training = []
        self.f1_value_validation = []
        self.f1_value_test = []

        self.results_folder = results_folder
        utilities.create_folder(self.results_folder)

        self.check_activation_fncts_length()

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
                activation=self.activation_fncts[-1],
                name='classifier_network_output_layer')(hidden_layer)

        else:
            #   output layer: since it is a softmax layer a neuron is needed to encode each class
            output_layer = Dense(
                self.number_of_classes,
                activation=self.activation_fncts[-1],
                name='classifier_network_output_layer')(input_layer)

        #   create the network model
        classifier_network_model = Model(inputs=input_layer,
                                         outputs=output_layer,
                                         name='classifier_network_model')

        if self.learning_rate == "None":
            classifier_network_model.compile(loss=self.loss,
                                             optimizer='adam',
                                             metrics=self.metrics)

        else:
            opt = optimizers.Adam(lr=float(self.learning_rate))
            classifier_network_model.compile(loss=self.loss,
                                             optimizer=opt,
                                             metrics=self.metrics)

        print(classifier_network_model.summary(line_length=100))

        return classifier_network_model

    def train_classifier_net(self,
                             training_data,
                             validation_data,
                             test_data=None,
                             save_per_epoch=False):

        training_set, training_supervision = utilities.from_input_to_data(training_data)
        validation_set, validation_supervision = utilities.from_input_to_data(validation_data)

        self.input_x_dimension = training_set.shape[1]
        self.number_of_classes = training_supervision.shape[1]

        test_set, test_supervision = None, None
        do_test = False
        if test_data is not None:
            do_test = True
            test_set, test_supervision = utilities.from_input_to_data(test_data)

        log_file = open(self.results_folder + "/log_file.txt", "a")

        epochs = int(self.epochs)

        batch_size = int(self.batch_size)

        perc_gpu = float(self.perc_gpu)
        gpu_setup.gpu_setup_v1(id_gpu=self.id_gpu, memory_percentage=perc_gpu)

        classifier_net_model = self.build_simple_network()

        for epoch in range(epochs):
            print("\n\n\nEpoch " + str(epoch))
            log_file.write("\n\n\nEpoch " + str(epoch))
            history_classifier_net = classifier_net_model.fit(x=training_set,
                                                              y=training_supervision,
                                                              batch_size=batch_size,
                                                              epochs=1,
                                                              shuffle=True,
                                                              validation_data=(validation_set, validation_supervision))

            res = history_classifier_net.history.get('loss')[0]
            self.classifier_network_loss_vec.append(res)
            log_file.write("\nClassifier loss ---> " + str(res))

            res = []
            for metric in self.metrics:
                res.append(history_classifier_net.history.get(metric)[0])
            self.classifier_network_metrics_vec.append(res)
            log_file.write("\nClassifier metrics ---> " + str(res))

            res = history_classifier_net.history.get('val_loss')[0]
            self.classifier_network_val_loss_vec.append(res)
            log_file.write("\nClassifier validation loss ---> " + str(res))

            res = []
            for metric in self.metrics:
                res.append(history_classifier_net.history.get('val_' + metric)[0])
            self.classifier_network_val_metrics_vec.append(res)
            log_file.write("\nClassifier validation metrics ---> " + str(res))

            if do_test:
                #   evaluation over the test set
                test_eval = classifier_net_model.evaluate(x=test_set, y=test_supervision, batch_size=batch_size)
                self.classifier_network_evaluation_on_test_set_loss_vec.append(
                    test_eval[0]
                )
                self.classifier_network_evaluation_on_test_set_metrics_vec.append(
                    test_eval[1]
                )

            #  these operations needs prediction and argmax transformation
            training_set_classes_supervision = np.argmax(training_supervision, axis=1)
            training_set_classes_prediction = np.argmax(
                classifier_net_model.predict(x=training_set, batch_size=batch_size), axis=1)

            validation_set_classes_supervision = np.argmax(validation_supervision, axis=1)
            validation_set_classes_prediction = np.argmax(
                classifier_net_model.predict(x=validation_set, batch_size=batch_size), axis=1)

            test_set_classes_supervision, test_set_classes_prediction = None, None
            if do_test:
                test_set_classes_supervision = np.argmax(test_supervision, axis=1)
                test_set_classes_prediction = np.argmax(
                    classifier_net_model.predict(x=test_set, batch_size=batch_size), axis=1)

            training_precision = utilities.compute_precision(y_classes=training_set_classes_supervision,
                                                             y_pred_classes=training_set_classes_prediction)
            log_file.write("\nClassifier training_precision ---> " + str(training_precision))

            training_recall = utilities.compute_recall(y_classes=training_set_classes_supervision,
                                                       y_pred_classes=training_set_classes_prediction)
            log_file.write("\nClassifier training_recall ---> " + str(training_recall))

            training_f1 = utilities.compute_f1_score(y_classes=training_set_classes_supervision,
                                                     y_pred_classes=training_set_classes_prediction)
            log_file.write("\nClassifier training_f1 ---> " + str(training_f1))

            self.f1_value_training.append(training_f1)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%

            validation_precision = utilities.compute_precision(y_classes=validation_set_classes_supervision,
                                                               y_pred_classes=validation_set_classes_prediction)
            log_file.write("\nClassifier validation_precision ---> " + str(validation_precision))

            validation_recall = utilities.compute_recall(y_classes=validation_set_classes_supervision,
                                                         y_pred_classes=validation_set_classes_prediction)
            log_file.write("\nClassifier validation_recall ---> " + str(validation_recall))

            validation_f1 = utilities.compute_f1_score(y_classes=validation_set_classes_supervision,
                                                       y_pred_classes=validation_set_classes_prediction)
            log_file.write("\nClassifier validation_f1 ---> " + str(validation_f1))

            self.f1_value_validation.append(validation_f1)

            if do_test:
                self.f1_value_test.append(utilities.compute_f1_score(y_classes=test_set_classes_supervision,
                                                                     y_pred_classes=test_set_classes_prediction))

            ####################################################################################################################

            #   save all vectors
            with open(self.results_folder + '/classifier_network_loss_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_loss_vec, f)
            with open(self.results_folder + '/classifier_network_categ_acc_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_metrics_vec, f)
            with open(self.results_folder + '/classifier_network_val_loss_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_val_loss_vec, f)
            with open(self.results_folder + '/classifier_network_val_categ_acc_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_val_metrics_vec, f)

            if do_test:
                with open(self.results_folder + 'classifier_network_evaluation_on_test_set_loss_vec.pkl', 'wb') as f:
                    pickle.dump(self.classifier_network_evaluation_on_test_set_loss_vec, f)
                with open(self.results_folder + 'classifier_network_evaluation_on_test_set_accuracy_vec.pkl', 'wb') as f:
                    pickle.dump(self.classifier_network_evaluation_on_test_set_loss_vec, f)

            with open(self.results_folder + '/f1_value_training_vec.pkl', 'wb') as f:
                pickle.dump(self.f1_value_training, f)
            with open(self.results_folder + '/f1_value_validation_vec.pkl', 'wb') as f:
                pickle.dump(self.f1_value_validation, f)

            if do_test:
                with open(self.results_folder + 'f1_value_test_vec.pkl', 'wb') as f:
                    pickle.dump(self.f1_value_test, f)

            if save_per_epoch:
                classifier_net_model.save(filepath=self.results_folder + "/classifier_net_model_epoch" + str(epoch))
                classifier_net_model.save_weights(
                    filepath=self.results_folder + "/classifier_net_model_weights_epoch" + str(epoch))

        if not save_per_epoch:
            classifier_net_model.save(filepath=self.results_folder + "/classifier_net_model")
            classifier_net_model.save_weights(
                filepath=self.results_folder + "/classifier_net_model_weights")

        log_file.close()
        return None
