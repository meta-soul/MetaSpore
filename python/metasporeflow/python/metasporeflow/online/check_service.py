import requests
import time
import traceback

def notifyRecommendService(host, port):
    print("notify recommend service %s:%s" % (host, port))
    max_wait = 300
    num = max_wait
    last_exception = None
    while num > 0:
        try:
            resp = requests.post('http://%s:%s/actuator/refresh' % (host, port))
        except Exception as ex:
            resp = None
            last_exception = ex
        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as ex:
                data = None
                last_exception = ex
            if data is not None:
                # succeed: print and return
                print(data)
                return
        print("retry refresh recommend service! %s:%s" % (host, port))
        time.sleep(1)
        num -= 1
    if last_exception is not None:
        traceback.print_exception(last_exception)
    message = "fail to notify recommend service %s:%s after waiting %d seconds" % (host, port, max_wait)
    raise RuntimeError(message)

if __name__ == "__main__":
    print("test")
    notifyRecommendService()
