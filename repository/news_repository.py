import datetime

import pandas as pd
from sqlalchemy import select

from model.news import News
from repository.base_repository import BaseRepository


class NewsRepository(BaseRepository):

    def save_news_data(self, news: News):

        self.session.add(news)

        return news

    def get_news_data(self, news_url: str):

        stmt = select(News).where(News.news_url == news_url)
        res = self.session.execute(stmt)
        res = res.scalar()
        return res

    def get_news_dataset(self) -> pd.DataFrame:
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        stmt = select(News).where(News.date_time < yesterday)
        all_news = self.session.execute(stmt)
        all_news = all_news.scalars().all()
        df = pd.DataFrame([item.__dict__ for item in all_news])
        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df.sort_values(by=["date_time"])
        return df
