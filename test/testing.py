import requests
from bs4 import BeautifulSoup

from model import stock


def get_header():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.183 Safari/537.36 Vivaldi/1.96.1147.47",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    return headers


page = 1
stock = "삼성전자"
base_url = f"https://search.daum.net/search?w=news&nil_search=btn&DA=NTB&enc=utf8&cluster=y&cluster_page=1&q={stock}&p={page}"
is_next_page = True
response = requests.get(base_url, headers=get_header())
list_soup = BeautifulSoup(response.text, "html.parser")
# 리스트 전체 가져오기
# 리스트에서 필요한거 뽑아서 객체 생성해저 집어넣기
# 리스트 반환하기
news_datas = list_soup.find("ul", class_="c-list-basic")
bs_list = news_datas.find_all("li") if news_datas else []
news_list = []
for item in bs_list:
    print(item)
    news_list.append(item)

# print(news_list)
