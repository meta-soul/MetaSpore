from aiohttp import web
from utils.log_util import get_logger
import os


class Tracking:
    def __init__(self):
        _when = os.environ.get('UPLOAD_WHEN')
        _interval = os.environ.get('UPLOAD_INTERVAL')
        _backup_count = os.environ.get('UPLOAD_BACKUP_COUNT')
        self.port = os.environ.get('PORT')
        self.logger = get_logger(_when, _interval, _backup_count)

    async def handleRequest(self, request):
        data = await request.json()
        self.logger.info(data)
        return web.Response(text='OK')

    def run(self):
        app = web.Application()
        app.router.add_route('POST', '/tracking', self.handleRequest)
        web.run_app(app, port=self.port)

    def check_request(self, request_data):
        if not request_data:
            return False, 'empty request data'
        if not isinstance(request_data, dict):
            return False, 'request data must be dict'
        if 'request_id' not in request_data:
            return False, 'app_id is required'
        if 'event_type' not in request_data:
            return False, 'version is required'
        if 'timestamp' not in request_data:
            return False, 'timestamp is required'
        if 'data' not in request_data:
            return False, 'data is required'
        return True, 'ok'


if __name__ == '__main__':
    tracking = Tracking()
    tracking.run()
