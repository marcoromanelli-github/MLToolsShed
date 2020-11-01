[![build_package](https://github.com/marcoromanelli-github/MLToolsShed/workflows/build_package/badge.svg)](https://github.com/marcoromanelli-github/MLToolsShed/actions)
[![Documentation](https://img.shields.io/badge/Documentation-yes-blue)](https://img.shields.io/badge/Documentation-yes-blue)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://img.shields.io/badge/License-MIT-yellow.svg)

# MLToolsShed
<a href="https://github.com/marcoromanelli-github/MLToolsShed"><img src="https://github-link-card.s3.ap-northeast-1.amazonaws.com/marcoromanelli-github/MLToolsShed.png" width="460px"></a>

A set of tools to make programming with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) more high level.

### Intallation
1. Install required packages by running
```console
foo@bar:~$ python -m pip install -U -r <path>/requirements.txt
```
2. Download the deplyed package as the artifact created by the last action in [Actions](https://github.com/marcoromanelli-github/MLToolsShed/actions) and unzip it.
3. Run the command below
```console
foo@bar:~$ python -m pip install -U <path>/built_pkg/<filename>.whl
```

### Current package structure
```
Package
├── MLToolsShed
│   ├── simple_network_manager_pkg
│   │   └── simple_network_manager.py
│   └── utilities_pkg
│       ├── gpu_setup.py
│       ├── read_CLI_options.py
│       ├── runtime_error_handler.py
│       └── utilities.py
├── MLToolsShed_test
│   ├── __init__.py
│   ├── test_0.py
│   ├── test_1.py
├── README.md
├── requirements.txt
└── setup.py
```

### How to
The folder [MLToolsShed_test](https://github.com/marcoromanelli-github/MLToolsShed/tree/master/MLToolsShed_test) contains examples which are useful to understand how to use the library.

#### read_CLI_options
This is a function which might come in handy when launching heavy experiments from the CMI. It reads all the parameters needed to build, tune and train a simple network model. For instance one could write the following script try_test.py
```python
from MLToolsShed_test import test_0, test_1


def run_test():
    test_0.test_0()
    #   test_1.test_1()


if __name__ == '__main__':
    run_test()

```
Supposing this command is launched from the CLI as
```console
foo@bar:~$ python try_test.py -lr 0.001
```
the output would be
```console
foo@bar:~$ {'learning_rate': 0.001}
```
After building the dictionary all the arguments can be directly passed to the handler for building, tuning and training the network.
The options are:
| Option     | Description                                                                    |
| ------     | -----------                                                                    |
| -mn        | Model name.                                                                    |
| -hnc       | List without spaces of the number of neurons per hidden layer, e.g. [10,20,30].|
| -lr        | Learning rate.                                                                 |
| -e         | Number of epochs.                                                              |
| -bs        | Batch size.                                                                    |
| --id_gpu   | Integer identifier of the GPU that we intend to use.                           |
| --perc_gpu | Percentage of the GPU memory to be used.                                       |
| -tr        | Absolute path to the training data.                                            |
| -val       | Absolute path to the validation data.                                          |
| -ts        | Absolute path to the test data.                                                |
| -actf      | List of activation functions without spaces, e.g. [relu,relu,relu,softmax].    |
| -metr      | List of metric functions without spaces, e.g. [categorical_accuracy].          |
| -loss      | Loss function for the training.                                                |
| -resdir    | Absolute path to the directory where the results will be stored.               |
