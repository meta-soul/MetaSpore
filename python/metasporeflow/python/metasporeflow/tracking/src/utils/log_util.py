def get_log_handler():
    import logging.handlers as handlers
    # from concurrent_log_handler import ConcurrentRotatingFileHandler
    logHandler = handlers.TimedRotatingFileHandler('tracking.log', when='S', interval=5, backupCount=5)
    return logHandler

def get_logger():
    import logging
    logger = logging.getLogger('tracking')
    logger.setLevel(logging.INFO)
    # when 是一个字符串的定义如下：
    # “S”: Seconds
    # “M”: Minutes
    # “H”: Hours
    # “D”: Days
    # “W”: Week
    # day(0 = Monday)
    #
    # interval
    # 是指等待多少个单位when的时间后，Logger会自动重建文件，当然，这个文件的创建
    # 取决于filename + suffix，若这个文件跟之前的文件有重名，则会自动覆盖掉以前的文件，所以
    # 有些情况suffix要定义的不能因为when而重复。
    #
    # backupCount
    # 是保留日志个数。默认的0是不会自动删除掉日志。若设10，则在文件的创建过程中
    # 库会判断是否有超过这个10，若超过，则会从最先创建的开始删除。
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logHandler = get_log_handler()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    return logger