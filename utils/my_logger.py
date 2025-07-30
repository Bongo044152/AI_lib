# file: utils/logger.py
# initialize logger for formatted output
# we choice module logging provided by python official
# logging module: https://docs.python.org/3/library/logging.html
# see more example: https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook

import logging

logger_flag = True

# this function is expected to run only once
def init_logger() -> None:
    """
    this function used to initialize logger, and not expect to be
    call twice.

    Returns:
        None: the same as null
    """

    # this function is expected to run only once
    global logger_flag
    if not logger_flag:
        return
    logger_flag = True

    # base configuretion
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s: [%(levelname)s] - %(message)s',
    )

    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)

    # initialize logger here
