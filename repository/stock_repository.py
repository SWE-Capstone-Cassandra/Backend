import datetime

from datetime import datetime

from sqlalchemy import select
from model.stock import Stock
from config import get_session, get_engine, Base


class StockRepository:

    def __init__(self) -> None:
        self.session = get_session()
        self.engine = get_engine()

    def get_stock_data_by_date(self, date: int, time: int) -> int:
        session = self.session()
        stmt = select(Stock).where(Stock.date == date).where(Stock.time == time)
        res = session.execute(stmt)
        return res.scalar().price

    def save_stock_data(self, time: int, price: int):
        stock_data = Stock()
        stock_data.date = int(datetime.today().date().strftime("%Y%m%d"))
        stock_data.time = time
        stock_data.price = price

        session = self.session()
        session.add(stock_data)
        session.commit()
        return price
