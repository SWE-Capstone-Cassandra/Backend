from datetime import datetime

from repository.stock_repository import StockRepository
from service.base_service import BaseService
from service.news_service import NewsService


class StockService(BaseService):

    def get_stock_data_by_date(self, date: int, time: int):
        return StockRepository(session=self.session).get_stock_data_by_date(date=date, time=time)

    def save_stock_data(self, time: int, price: int):
        price = StockRepository(session=self.session).save_stock_data(time=time, price=price)

        return True
