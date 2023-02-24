# examples/server_simple.py
from aiohttp import web


class Tracking:
    def __init__(self):
        self.logger = self.init_timed_rotating_file()

    async def handleRequest(self, request):
        data = await request.json()
        self.logger.info(data)
        return web.Response(text='OK')

    def run(self):
        app = web.Application()

        app.router.add_route('POST', '/tracking', self.handleRequest)

        web.run_app(app, port=50000)

    def init_timed_rotating_file(self):
        import logging
        import logging.handlers as handlers
        import time
        logger = logging.getLogger('tracking')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logHandler = handlers.TimedRotatingFileHandler('tracking.log', when='M', interval=1, backupCount=0)
        logHandler.setLevel(logging.INFO)
        logHandler.setFormatter(formatter)
        logger.addHandler(logHandler)
        return logger


if __name__ == '__main__':
    tracking = Tracking()
    tracking.run()
