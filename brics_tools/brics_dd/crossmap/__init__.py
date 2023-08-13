import logging
from brics_tools.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
crossmap_logger = logging.getLogger("crossmap_logger")
crossmap_logger.info("Initiating crossmap_logger.")
