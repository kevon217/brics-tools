import logging
from brics_tools.utils.logger.config_logging import setup_log, log, copy_log
from brics_tools.utils import helper

# CREATE LOGGER
setup_log()
logger = logging.getLogger("studysearch_app_logger")
logger.info("Initiating studysearch_app_logger")


__all__ = ["logger", "log", "copy_log", "helper"]
