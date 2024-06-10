import datetime

import pandas as pd
from sqlalchemy import select

from model.stock import Stock
from repository.base_repository import BaseRepository


class StockRepository(BaseRepository):

    def get_stock_data_by_date(self, date_time: datetime) -> int:

        stmt = select(Stock).where(Stock.date_time == date_time)
        res = self.session.execute(stmt)
        res = res.scalar()
        if res:
            return res.price
        else:
            return 0

    def get_stock_data_now(self) -> int:

        stmt = select(Stock).order_by(Stock.date_time.desc()).limit(1)
        res = self.session.execute(stmt)
        res = res.scalars().all()
        return res[0].price

    def save_stock_data(self, price: int):
        stock_data = Stock()

        stock_data.date_time = datetime.datetime.now().replace(second=0, microsecond=0)
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
