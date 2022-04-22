import requests

TXT2IMG_SEARCH_SERVICE_URL = 'http://172.31.0.197:8084/img/search'
TXT2TXT_SEARCH_SERVICE_URL = 'http://172.31.0.197:8082/txt/search'
IMG2IMG_SEARCH_SERVICE_URL = 'http://172.31.0.197:8083/search'
IMG2TXT_SEARCH_SERVICE_URL = 'http://172.31.0.197:8083/classify'


def txt2img_search_service(query, top_k=10):
    params = {'query': query, 'k': top_k}
    res = requests.get(TXT2IMG_SEARCH_SERVICE_URL, params=params)
    #print(res.text)
    res = res.json()
    if res['errno'] != 0:
        return []
    return [{'title': x['name'], 'content': '<img src="{}" />'.format(x['url']), 'url': x['url'], 'score': x['score']} for x in res['data']]

def txt2txt_search_service(query, top_k=10):
    params = {'query': query, 'k': top_k}
    res = requests.get(TXT2TXT_SEARCH_SERVICE_URL, params=params).json()
    #print(res)
    #if ['errno'] != 0:
    #    return []
    return [{'title': x['title'], 'content': x['content'], 'url': '', 'score': x['score']} for x in res['data']]

def img2img_search_service(img, top_k=10):
    params = {'k': top_k}
    files = {'image': img}
    r = requests.post(IMG2IMG_SEARCH_SERVICE_URL, data=params, files=files)
    res = r.json()
    return [{'title': x['name'], 'content': '<img src="{}" />'.format(x['url']), 'url': x['url'], 'score': x['score']} for x in res['data']]

def img2txt_search_service(img, top_k=10):
    files = {'image': img}
    r = requests.post(IMG2TXT_SEARCH_SERVICE_URL, files=files)
    res = r.json()
    return [{'title': x['label_name_en'], 'content': x['label_name_zh'], 'url': '', 'score': x['score']} for x in res['data']]

if __name__ == '__main__':
    print(txt2img_search_service("猫"))
    print(txt2img_search_service("大象"))
    print(txt2txt_search_service("拉肚子怎么办？"))
