import logging
import os

# logging level for dynamo generated graphs/bytecode/guards
INFO_CODE = 15


# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers():
    return [
        logging.getLogger("torchdynamo"),
        logging.getLogger("torchinductor"),
    ]


# Set the level of all loggers that torchdynamo is responsible for
def set_loggers_level(level):
    for logger in get_loggers():
        logger.setLevel(level)


LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "torchdynamo_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "torchdynamo_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "torchdynamo_format",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "torchdynamo": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
        "torchinductor": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
    },
    "disable_existing_loggers": False,
}


# initialize torchdynamo loggers
def init_logging(log_level, log_file_name=None):
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.config.dictConfig(LOGGING_CONFIG)
        if log_file_name is not None:
            log_file = logging.FileHandler(log_file_name)
            log_file.setLevel(log_level)
            for logger in get_loggers():
                logger.addHandler(log_file)

    set_loggers_level(log_level)
