from urllib.parse import urlparse


def url_without_query_string(url):
    o = urlparse(url)
    return o.scheme + "://" + o.netloc + o.path


if __name__ == "__main__":
    urls = [
        "https://stackoverflow.com/questions/7990301?aaa=aaa",
        "https://stackoverflow.com/questions/7990300?fr=aladdin",
        "https://stackoverflow.com/questions/22375#6",
        "https://stackoverflow.com/questions/22375?",
        "https://stackoverflow.com/questions/22375#3_1",
        "https://www.trustpilot.com/categories/clothing_store?numberofreviews=0&page=72&status=all&timeperiod=0",
    ]
    for url in urls:
        print(url_without_query_string(url))
