import requests

TXT2TXT_SEARCH_SERVICE_URL = 'http://localhost:8080/qa/user/10'
TXT2IMG_SEARCH_SERVICE_URL = 'http://localhost:8080/t2i/user/10'

IMG2IMG_SEARCH_SERVICE_URL = 'http://172.31.0.197:8083/search'
IMG2TXT_SEARCH_SERVICE_URL = 'http://172.31.0.197:8083/classify'


def txt2txt_search_service(query, top_k=10):
    params = {'query': query}
    res = requests.post(TXT2TXT_SEARCH_SERVICE_URL, json=params).json()
    items = []
    if not res.get('searchItemModels'):
        return items
    for item in res['searchItemModels'][0]:
        items.append({
            'title': item['summary']['question'],
            'content': item['summary']['answer'],
            'url': '',
            'score': item['score']
        })
    return items

def txt2img_search_service(query, top_k=10):
    # not impl
    return []
    params = {'query': query}
    res = requests.post(TXT2IMG_SEARCH_SERVICE_URL, json=params)
    #print(res.text)
    res = res.json()
    return res
    if res['errno'] != 0:
        return []
    return [{'title': x['name'], 'content': '<img src="{}" />'.format(x['url']), 'url': x['url'], 'score': x['score']} for x in res['data']]


def img2img_search_service(img, top_k=10):
    # not impl
    return []
    params = {'k': top_k}
    files = {'image': img}
    r = requests.post(IMG2IMG_SEARCH_SERVICE_URL, data=params, files=files)
    res = r.json()
    return [{'title': x['name'], 'content': '<img src="{}" />'.format(x['url']), 'url': x['url'], 'score': x['score']} for x in res['data']]

def img2txt_search_service(img, top_k=10):
    # not impl
    return []
    files = {'image': img}
    r = requests.post(IMG2TXT_SEARCH_SERVICE_URL, files=files)
    res = r.json()
    return [{'title': x['label_name_en'], 'content': x['label_name_zh'], 'url': '', 'score': x['score']} for x in res['data']]

if __name__ == '__main__':
    print(txt2txt_search_service("拉肚子怎么办？"))
    #print(txt2img_search_service("猫"))
    #print(txt2img_search_service("大象"))
