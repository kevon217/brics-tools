import logging
from brics_tools.utils.logger.config_logging import setup_log, log, copy_log

# CREATE LOGGER
setup_log()
logger = logging.getLogger("query_engine_logger")
logger.info("Initiating query_engine_logger")
