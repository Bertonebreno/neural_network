import logging

logging.basicConfig(level=logging.INFO)
fmt = "%(name)s[%(process)d] %(levelname)s %(message)s"

logger = logging.getLogger("interpreter")
logger.setLevel(logging.INFO)  # defaults to WARN
