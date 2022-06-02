import logging

log_file = 'log.txt'
log_level = logging.DEBUG

logger = logging.getLogger(__name__)
logger.setLevel(log_level)

file_handler = logging.FileHandler(filename=log_file)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
file_handler.setLevel(log_level)
stream_handler.setLevel(log_level)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info('abc')
logger.debug('debug')
logger.error('error')
