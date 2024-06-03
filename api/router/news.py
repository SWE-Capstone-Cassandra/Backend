from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import get_session
from schema.news_schema import NewsListAtt, NewsResponse
from service.news_service import NewsService

news_router = APIRouter()


@news_router.get("/{news_id}")
def get_news_by_news_id(news_id: int, session: Session = Depends(get_session)) -> NewsResponse:
    res = NewsService(session=session).get_news_data_by_id(news_id=news_id)

    return res


@news_router.get("/list")
def get_news_list_by_name(session: Session = Depends(get_session)) -> List[NewsListAtt]:
    return NewsService(session=session).get_news_list()
