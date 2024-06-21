import datetime
from typing import List

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import load_only

from model.news import News
from repository.base_repository import BaseRepository
from utils.enum.stock_code import StockCode


class NewsRepository(BaseRepository):

    def save_news_data(self, news: News):

        self.session.add(news)
        self.session.flush()
        return news

    def get_news_data(self, news_url: str):

        stmt = select(News).where(News.news_id == news_url)
        res = self.session.execute(stmt)
        res = res.scalar()
        return res

    def get_news_data_by_id(self, news_id: int) -> News:
        stmt = select(News).where(News.news_id == news_id)
        res = self.session.execute(stmt)
        res = res.scalar()
        return res

    def get_news_dataset_by_stock_code(self, stock_code: StockCode) -> pd.DataFrame:
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        stmt = select(News).where(News.stock_code == stock_code).where(News.stock_code == stock_code).where(News.date_time < yesterday)
        all_news = self.session.execute(stmt)
        all_news = all_news.scalars().all()
        df = pd.DataFrame([item.__dict__ for item in all_news])
        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df.sort_values(by=["date_time"])
        return df

    def get_news_list_by_stock_code(self, stock_code: StockCode) -> List[News]:
        # stmt = select(News).order_by(News.date_time.desc()).limit(10)
        stmt = (
            select(News)
            .where(News.stock_code == stock_code)
            .order_by(News.date_time.desc())
            .limit(10)
            .options(load_only(News.news_id, News.title))
        )
        res = self.session.execute(stmt)
        res = res.scalars().all()

        return res

    def get_news_id_by_period(self, start_day, end_day, stock_code: StockCode) -> List:
        stmt = (
            select(News)
            .where(News.stock_code == stock_code)
            .where(News.date_time < end_day)
            .where(News.date_time > start_day)
            .options(load_only(News.news_id))
        )
        res = self.session.execute(stmt)
        res = res.scalars().all()
        return res

    def get_news_by_period(self, start_day, end_day, stock_code: StockCode) -> List:
        stmt = select(News).where(News.stock_code == stock_code).where(News.date_time < end_day).where(News.date_time > start_day)
        res = self.session.execute(stmt)
        res = res.scalars().all()
        return res

    def get_news_total_page_by_stock_code(self, stock_code: StockCode) -> int:
        stmt = select(func.count()).select_from(News).where(News.stock_code == stock_code)
        res = self.session.execute(stmt)

        res = res.scalar()

        return res
