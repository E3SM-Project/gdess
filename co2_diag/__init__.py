
def _config_logger():
    """Configure the root logger"""
    import os
    import logging.config
    import json

    logconfig_path = os.path.dirname(os.path.realpath(__file__)) + '/config/log_config.json'
    with open(logconfig_path, 'r') as logging_configuration_file:
        config_dict = json.load(logging_configuration_file)

    logging.config.dictConfig(config_dict)


def _change_log_level(a_logger, level):
    a_logger.setLevel(level)
    for handler in a_logger.handlers:
        handler.setLevel(level)


_config_logger()
