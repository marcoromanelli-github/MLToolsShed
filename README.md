[![build_package](https://github.com/marcoromanelli-github/MLToolsShed/workflows/build_package/badge.svg)](https://github.com/marcoromanelli-github/MLToolsShed/actions)
[![Documentation](https://img.shields.io/badge/Dcoumentation-yes-blue)](https://img.shields.io/badge/Dcoumentation-yes-blue)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://img.shields.io/badge/License-MIT-yellow.svg)

# MLToolsShed
A set of tools to make programming with TensorFlow and Keras more high level.

### Intallation
1. Install required packages by running
```console
foo@bar:~$ python -m pip install -U -r <path>/requirements.txt
```
2. Download the deplyed package as the artifact created by the last action in [Actions](https://github.com/marcoromanelli-github/MLToolsShed/actions) and unzip it.
3. Run
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
├── test_folder
│   ├── __init__.py
│   ├── test_0.py
│   ├── test_1.py
├── README.md
├── required_pkg.txt
└── setup.py
```

### How to
The folder [test_folder](https://github.com/marcoromanelli-github/MLToolsShed/tree/master/test_folder) contains examples which are useful to 
