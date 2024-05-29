import datetime

import pandas as pd
from sqlalchemy import select

from model.stock import Stock
from repository.base_repository import BaseRepository


class StockRepository(BaseRepository):

    def get_stock_data_by_date(self, date_time: datetime) -> int:

        stmt = select(Stock).where(Stock.date_time == date_time)
        res = self.session.execute(stmt)
        return res.scalar().price

    def save_stock_data(self, time: int, price: int):
        stock_data = Stock()
        date = datetime.datetime.today().date().strftime("%Y%m%d")
        time = str(time)
        stock_data.date_time = datetime.datetime.strptime(date + time, "%Y%m%d%H%M")
        stock_data.price = price

        self.session.add(stock_data)
        return price

    def get_stock_dataset(self):
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        stmt = select(Stock).where(Stock.date_time < yesterday)
        all_news = self.session.execute(stmt)
        all_news = all_news.scalars().all()
        df = pd.DataFrame([item.__dict__ for item in all_news])
        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df.sort_values(by=["date_time"])
        return df
