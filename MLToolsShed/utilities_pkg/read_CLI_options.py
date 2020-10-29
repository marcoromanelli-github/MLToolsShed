import sys
from MLToolsShed.utilities_pkg import runtime_error_handler

MODEL_NAME = ""
HIDDEN_NEAURONS_CARD = []
EPOCHS = None
BATCH_SIZE = None
ID_GPU = None
PERC_GPU = None
TRAINING_SET_PATH = ""
VALIDATION_SET_PATH = ""
TEST_SET_PATH = ""


def read_command_line_options():
    thismodule = sys.modules[__name__]

    for idx, key_val in enumerate(sys.argv, 0):
        if key_val in ['--model_name', '-mn'] and len(sys.argv) > idx + 1:
            try:
                thismodule.MODEL_NAME = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--learning_rate', '-lr'] and len(sys.argv) > idx + 1:
            try:
                thismodule.LEARNING_RATE = float(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--training_set_path', '-tr'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TRAINING_SET_PATH = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--validation_set_path', '-val'] and len(sys.argv) > idx + 1:
            try:
                thismodule.VALIDATION_SET_PATH = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--test_set_path', '-ts'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_SET_PATH = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--hidden_neurons_card', '-hnc'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print(string_to_be_adapted)
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                HIDDEN_NEAURONS_CARD_ = []
                for item in split_list:
                    try:
                        HIDDEN_NEAURONS_CARD_.append(int(item))
                    except ValueError as val_err:
                        runtime_error_handler.exception_call(idx=idx, key_val=key_val)
                thismodule.HIDDEN_NEAURONS_CARD = HIDDEN_NEAURONS_CARD_

        if key_val in ['--epochs', '-e'] and len(sys.argv) > idx + 1:
            try:
                thismodule.EPOCHS = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--batch_size', '-bs'] and len(sys.argv) > idx + 1:
            if sys.argv[idx + 1].strip() == 'all' or sys.argv[idx + 1].strip() == "ALL":
                thismodule.BATCH_SIZE = None
            else:
                try:
                    thismodule.BATCH_SIZE = int(sys.argv[idx + 1].strip())
                except ValueError as val_err:
                    runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--id_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.ID_GPU = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)

        if key_val in ['--perc_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.PERC_GPU = float(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                runtime_error_handler.exception_call(idx=idx, key_val=key_val)
