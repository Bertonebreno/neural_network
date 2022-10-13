import logging

logging.basicConfig(level=logging.INFO)
fmt = "%(name)s[%(process)d] %(levelname)s %(message)s"

logger = logging.getLogger("neural_network")
logger.setLevel(logging.INFO)  # defaults to WARN
