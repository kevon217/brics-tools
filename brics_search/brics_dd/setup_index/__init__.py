import logging
from brics_search.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
setup_logger = logging.getLogger("setup_index_logger")
setup_logger.info("Initiating setup_index_logger logger.")
