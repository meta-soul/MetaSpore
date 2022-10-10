import requests
import time

def notifyRecommendService(host, port):
    print("notify recommend service: %s:%s" % (host, port))
    num = 14
    while num > 0:
        resp = requests.post('http://%s:%s/actuator/refresh' % (host, port))
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(data)
            except:
                pass
            break
        print("retry refresh recommend service! %s:%s" % (host, port))
        time.sleep(1)
        num -= 1

if __name__ == "__main__":
    print("test")
    notifyRecommendService()
