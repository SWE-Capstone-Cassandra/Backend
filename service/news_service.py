import time
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from model.news import News
from repository.news_repository import NewsRepository
from service.base_service import BaseService
from service.cleaner import clean_text

CAPTCHA = "https://captcha.search.daum.net"


class NewsService(BaseService):

    def get_news_data(self, date_time: int, url: str) -> News:
        try:
            response = requests.get(url, headers=self.get_header())
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", class_="news_view fs_type1") if soup else ""
            title = soup.find("h3", class_="tit_view").text if soup else ""
            writer = soup.find("span", class_="txt_info").text if soup else ""
            content = clean_text(content.text)

            news_data = News()
            news_data.news_url = url
            news_data.date_time = datetime.strptime(str(date_time), "%Y%m%d%H%M")
            news_data.title = title
            news_data.writer = writer
            news_data.content = content
            return news_data
        except Exception as e:
            print(e)
            return None

    def save_news_data(self, news: News):
        res = NewsRepository(session=self.session).save_news_data(news=news)
        # prediction= 모델을 돌려서 나온 결과값.
        # prediction = PredictionRepository().save_news_prediction(news_prediction=prediction)

        return res

    def get_news_list(self, item_name: str, page: int):

        base_url = (
            f"https://search.daum.net/search?w=news&nil_search=btn&DA=PGD&enc=utf8&cluster=y&cluster_page=1&q={item_name}&p={page}"
        )

        response = requests.get(base_url, headers=self.get_header())
        list_soup = BeautifulSoup(response.text, "html.parser")

        if CAPTCHA in str(list_soup):
            response = requests.get(base_url, headers=self.get_header(), proxies=self.get_proxies())
            list_soup = BeautifulSoup(response.text, "html.parser")

        news_datas = list_soup.find("ul", class_="c-list-basic")
        bs_list = news_datas.find_all("li") if news_datas else []
        news_list = []
        for item in bs_list:
            news_list.append(item)

        return news_list

    def get_news_list_min(self, item_name: str, time_now) -> List:
        # 주소 리스트를 반환
        min_1 = 100
        base_url = f"https://search.daum.net/search?nil_suggest=btn&w=news&DA=STC&cluster=y&q={item_name}&sd={time_now-min_1}&ed={time_now}&period=u"
        print(base_url)
        response = requests.get(base_url, headers=self.get_header())
        list_soup = BeautifulSoup(response.text, "html.parser")

        if CAPTCHA in str(list_soup):
            response = requests.get(base_url, headers=self.get_header(), proxies=self.get_proxies())
            list_soup = BeautifulSoup(response.text, "html.parser")

        news_datas = list_soup.find("ul", class_="c-list-basic")
        div_tags = news_datas.find_all("p", class_="conts-desc clamp-g2") if news_datas else []
        url_list = []
        for tag in div_tags:
            a_tag = tag.find("a") if tag else None
            url_list.append(a_tag["href"] if a_tag else "No link found")

        return url_list

    def is_page(self, url, page):
        options = Options()
        options.add_argument("--disable-web-security")  # 웹 보안 비활성화
        options.add_argument("headless")
        options.headless = True

        driver = webdriver.Chrome(options=options)
        driver.get(url)

        # 페이지가 완전히 로드될 때까지 대기
        time.sleep(0.001)

        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # class="link_page"를 가진 모든 'a' 태그 찾기
        this_page = soup.find("em", class_="link_page")
        driver.quit()
        if this_page:
            # 찾은 링크 출력
            if str(this_page.text) == str(page):
                return True
            else:
                return False
        # WebDriver 종료
        else:
            return False

    def get_stock_by_item_code(self, item_code: str):
        pass

    # def get_

    def get_header(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.183 Safari/537.36 Vivaldi/1.96.1147.47",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        return headers

    def get_proxies(self):
        proxies = {"http": "socks5://127.0.0.1:9050", "https": "socks5://127.0.0.1:9050"}
        return proxies
