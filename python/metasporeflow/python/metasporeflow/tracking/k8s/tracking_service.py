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
        web.run_app(app, port=50001)

    def check_request(self, request_data):
        if not request_data:
            return False, 'empty request data'
        if not isinstance(request_data, dict):
            return False, 'request data must be dict'
        if 'request_id' not in request_data:
            return False, 'app_id is required'
        if 'event_type' not in request_data:
            return False, 'version is required'
        if 'data' not in request_data:
            return False, 'data is required'
        return True, 'ok'


if __name__ == '__main__':
    tracking = Tracking()
    tracking.run()
