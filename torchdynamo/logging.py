import logging
import logging.config
import os


# Return all loggers that torchdynamo is responsible for
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
def init_logging(level, file_name=None):
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.config.dictConfig(LOGGING_CONFIG)
        if file_name is not None:
            log_file = logging.FileHandler(file_name)
            log_file.setLevel(level)
            for logger in get_loggers():
                logger.addHandler(log_file)

    set_loggers_level(level)
