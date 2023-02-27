from aiohttp import web
from utils.log_util import get_logger


class Tracking:
    def __init__(self):
        self.logger = get_logger()

    async def handleRequest(self, request):
        data = await request.json()
        self.logger.info(data)
        return web.Response(text='OK')

    def run(self):
        app = web.Application()
        app.router.add_route('POST', '/tracking', self.handleRequest)
        web.run_app(app, port=50000)

if __name__ == '__main__':
    tracking = Tracking()
    tracking.run()
