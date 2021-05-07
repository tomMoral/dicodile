import os


def pytest_configure(config):
    # Set DICOD in debug mode
    os.environ["TESTING_DICOD"] = "1"
