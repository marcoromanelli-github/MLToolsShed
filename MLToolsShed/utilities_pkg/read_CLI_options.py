import sys
from MLToolsShed.utilities_pkg import runtime_error_handler

MODEL_NAME = ''
HIDDEN_NEAURONS_CARD = []
LEARNING_RATE = None
EPOCHS = None
BATCH_SIZE = None
ID_GPU = None
PERC_GPU = None
TRAINING_SET_PATH = ''
VALIDATION_SET_PATH = ''
TEST_SET_PATH = ''
ACTIVATION_FNCTS = []
METRICS_LIST = []
LOSS = ''
RESULT_FOLDER = ''


def read_command_line_options():
    thismodule = sys.modules[__name__]
    options_dict = {}

    for idx, key_val in enumerate(sys.argv, 0):
        if key_val in ['--model_name', '-mn'] and len(sys.argv) > idx + 1:
            try:
                thismodule.MODEL_NAME = sys.argv[idx + 1].strip()
                options_dict['model_name'] = MODEL_NAME
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--result_folder', '-resdir'] and len(sys.argv) > idx + 1:
            try:
                thismodule.MODEL_NAME = sys.argv[idx + 1].strip()
                options_dict['result_folder'] = RESULT_FOLDER
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--loss_function', '-loss'] and len(sys.argv) > idx + 1:
            try:
                thismodule.LOSS = sys.argv[idx + 1].strip()
                options_dict['loss_function'] = LOSS
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--learning_rate', '-lr'] and len(sys.argv) > idx + 1:
            try:
                thismodule.LEARNING_RATE = float(sys.argv[idx + 1].strip())
                options_dict['learning_rate'] = LEARNING_RATE
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--training_set_path', '-tr'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TRAINING_SET_PATH = sys.argv[idx + 1].strip()
                options_dict['training_set_path'] = TRAINING_SET_PATH
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--validation_set_path', '-val'] and len(sys.argv) > idx + 1:
            try:
                thismodule.VALIDATION_SET_PATH = sys.argv[idx + 1].strip()
                options_dict['validation_set_path'] = VALIDATION_SET_PATH
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--test_set_path', '-ts'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_SET_PATH = sys.argv[idx + 1].strip()
                options_dict['test_set_path'] = TEST_SET_PATH
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--hidden_neurons_card', '-hnc'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            # print(string_to_be_adapted)
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                # print('split_list', split_list)
                HIDDEN_NEAURONS_CARD_ = []
                for item in split_list:
                    try:
                        HIDDEN_NEAURONS_CARD_.append(int(item))
                    except ValueError as val_err:
                        runtime_error_handler.exception_call(idx=idx, key_val=key_val)
                thismodule.HIDDEN_NEAURONS_CARD = HIDDEN_NEAURONS_CARD_
            options_dict['hidden_neuron_card'] = HIDDEN_NEAURONS_CARD

        if key_val in ['--activation_fncts', '-actf'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            # print(string_to_be_adapted)
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                # print('split_list', split_list)
                ACTIVATION_FNCTS_ = []
                for item in split_list:
                    try:
                        ACTIVATION_FNCTS_.append(item)
                    except ValueError as val_err:
                        runtime_error_handler.exception_call(idx=idx, key_val=key_val)
                thismodule.ACTIVATION_FNCTS = ACTIVATION_FNCTS_
            options_dict['activation_fncts'] = ACTIVATION_FNCTS

        if key_val in ['--metrics_list', '-metr'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            # print(string_to_be_adapted)
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                # print('split_list', split_list)
                METRICS_LIST_ = []
                for item in split_list:
                    try:
                        METRICS_LIST_.append(item)
                    except ValueError as val_err:
                        runtime_error_handler.exception_call(idx=idx, key_val=key_val)
                thismodule.METRICS_LIST = METRICS_LIST_
            options_dict['metrics_list'] = METRICS_LIST

        if key_val in ['--epochs', '-e'] and len(sys.argv) > idx + 1:
            try:
                thismodule.EPOCHS = int(sys.argv[idx + 1].strip())
                options_dict['epopchs'] = EPOCHS
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--batch_size', '-bs'] and len(sys.argv) > idx + 1:
            if sys.argv[idx + 1].strip() == 'all' or sys.argv[idx + 1].strip() == "ALL":
                thismodule.BATCH_SIZE = None
            else:
                try:
                    thismodule.BATCH_SIZE = int(sys.argv[idx + 1].strip())
                    options_dict['batch_size'] = BATCH_SIZE
                except ValueError as val_err:
                    runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--id_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.ID_GPU = int(sys.argv[idx + 1].strip())
                options_dict['id_gpu'] = ID_GPU
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--perc_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.PERC_GPU = float(sys.argv[idx + 1].strip())
                options_dict['perc_gpu'] = PERC_GPU
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

    return options_dict
