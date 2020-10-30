import random
import numpy as np
from MLToolsShed.simple_network_manager_pkg import simple_network_manager


def test_1():
    training_data = np.zeros((1000, 10))
    validation_data = np.zeros((100, 10))
    test_data = np.zeros((500, 10))

    for j in range(training_data.shape[1]):
        for i in range(training_data.shape[0]):
            if j != training_data.shape[1] - 1:
                training_data[i, j] = random.uniform(0, 1)
            else:
                training_data[i, j] = random.randint(0, 1)

        for i in range(validation_data.shape[0]):
            if j != validation_data.shape[1] - 1:
                validation_data[i, j] = random.uniform(0, 1)
            else:
                validation_data[i, j] = random.randint(0, 1)

        for i in range(test_data.shape[0]):
            if j != test_data.shape[1] - 1:
                test_data[i, j] = random.uniform(0, 1)
            else:
                test_data[i, j] = random.randint(0, 1)

    simple_network_man = simple_network_manager.SimpleNetworkManager(id_gpu=0,
                                                                     perc_gpu=0.3,
                                                                     learning_rate=0.001,
                                                                     hidden_neurons_cards=[10, 20, 10],
                                                                     activation_fncts=['relu', 'relu', 'relu', 'softmax'],
                                                                     epochs=30,
                                                                     batch_size=100,
                                                                     results_folder='/tmp/results',
                                                                     metrics_list=['categorical_accuracy', 'accuracy'],
                                                                     loss='categorical_crossentropy')

    simple_network_man.train_classifier_net(training_data=training_data,
                                            validation_data=validation_data,
                                            test_data=test_data,
                                            save_per_epoch=False)
