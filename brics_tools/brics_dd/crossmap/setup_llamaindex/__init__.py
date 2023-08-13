import logging
from brics_tools.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
setup_index_logger = logging.getLogger("setup_index_logger")
setup_index_logger.info("Initiating setup_llamaindex logger.")
