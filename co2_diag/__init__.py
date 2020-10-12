import os
import logging.config
import json

# If applicable, delete the existing log file to generate a fresh log file during each execution
# if os.path.isfile("python_logging.log"):
#     os.remove("python_logging.log")

# Config the root logger
path = os.path.dirname(os.path.realpath(__file__)) + '/config/log_config.json'
with open(path, 'r') as logging_configuration_file:
    config_dict = json.load(logging_configuration_file)

logging.config.dictConfig(config_dict)


def _change_log_level(a_logger, level):
    a_logger.setLevel(level)
    for handler in a_logger.handlers:
        handler.setLevel(level)
