from typing import Optional
import requests

WIKI_SUMMARY = 'https://en.wikipedia.org/api/rest_v1/page/summary/{title}'

def wikipedia_summary(title: str) -> Optional[str]:
    try:
        url = WIKI_SUMMARY.format(title=title.replace(' ', '_'))
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            js = r.json()
            return js.get('extract')
    except Exception:
        pass
    return None

def numista_search_url(query: str) -> str:
    import requests as _rq
    return f"https://en.numista.com/catalogue/index.php?search={_rq.utils.quote(query)}"
