import requests

GOOGLE_CSE_CX = '012990202937599964354:5mjc-lnf-cw'
GOOGLE_API_KEY = 'AIzaSyAuZvOKQ-KXipXwkmZWpml0XIwtik_fY-I'


def get_related_snippet_via_google_search(query, max_num=10):
    """
    :param query: search query in string
    :param max_num: max number of search results [1,10]
    :return: list of URLs
    """
    params = {
        'q': query,
        'num': max_num,
        'start': 1,
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_CX
    }

    response = requests.get('https://www.googleapis.com/customsearch/v1', params)
    result_items = response.json().get('items')

    if result_items is None:
        return []
    else:
        return [item.get('snippet') for item in result_items]