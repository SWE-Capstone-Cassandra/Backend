import datetime
from datetime import datetime

from sqlalchemy import select

from model.stock import Stock
from repository.base_repository import BaseRepository


class StockRepository(BaseRepository):

    def get_stock_data_by_date(self, date: int, time: int) -> int:

        stmt = select(Stock).where(Stock.date == date).where(Stock.time == time)
        res = self.session.execute(stmt)
        return res.scalar().price

    def save_stock_data(self, time: int, price: int):
        stock_data = Stock()
        stock_data.date = int(datetime.today().date().strftime("%Y%m%d"))
        stock_data.time = time
        stock_data.price = price

        self.session.add(stock_data)
        return price
