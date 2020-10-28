import sys


def runtime_error_handler(str_, add):
    if str_ == "loss_mismatch":
        sys.exit("The number of input losses creates a mismatch in function " + str(add))
    if str_ == "folder_creation_failed":
        sys.exit("Folder creation failed in function " + str(add))
    if str_ == "unspecified_option":
        sys.exit("Unknown option in function " + str(add))


def exception_call(idx, key_val):
    sys.exit(sys.argv[idx + 1].strip() + " is not a valid argument for option " + key_val)
