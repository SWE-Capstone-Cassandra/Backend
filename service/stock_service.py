from datetime import datetime

from repository.stock_repository import StockRepository
from service.base_service import BaseService
from service.news_service import NewsService


class StockService(BaseService):

    def get_stock_data_by_date(self, date: int, time: int):
        return StockRepository(session=self.session).get_stock_data_by_date(date=date, time=time)

    def save_stock_data(self, time: int, price: int):
        price = StockRepository(session=self.session).save_stock_data(time=time, price=price)
        news_service = NewsService(session=self.session)
        time_now = (int(datetime.today().date().strftime("%Y%m%d")) * 10000 + time) * 100
        news_min_list = news_service.get_news_list_min(item_name="삼성전자", time_now=time_now)
        print(news_min_list)
        for news in news_min_list:
            data = news_service.get_news_data(time=time, url=news)
            print(data)
            res = news_service.save_news_data(news=data)
            print(res)
        return True
