from MLToolsShed.utilities_pkg import read_CLI_options


def test_0():
    options_dict = read_CLI_options.read_command_line_options()
    print(options_dict)


if __name__ == '__main__':
    test_0()
