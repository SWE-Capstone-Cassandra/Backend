from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import get_session
from service.news_service import NewsService

news_router = APIRouter()


@news_router.get("/{news_id}")
def get_news_by_news_id(news_id: str, session: Session = Depends(get_session)):
    res = NewsService(session=session).get_news_data(news_id=news_id)

    return res


@news_router.get("/list/{item_name}/{page}")
def get_news_list_by_name(item_name: str, page: int, session: Session = Depends(get_session)):
    return NewsService(session=session).get_news_list(item_name=item_name, page=page)
