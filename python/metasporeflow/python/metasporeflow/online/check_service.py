import requests
import time

def notifyRecommendService(host="127.0.0.1", port=8081):
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
        print("retry refresh recommend service!")
        time.sleep(1)
        num -= 1

if __name__ == "__main__":
    print("test")
    notifyRecommendService()
