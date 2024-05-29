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
