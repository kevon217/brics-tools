import logging
from brics_tools.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
utils_logger = logging.getLogger("helper_logger")
