import logging
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import get_session
from schema.news_schema import NewsListAtt, NewsResponse
from service.news_service import NewsService
from utils.enum.stock_code import StockCode

news_router = APIRouter()
logger = logging.getLogger("uvicorn")


@news_router.get("/{news_id}")
def get_news_by_news_id(news_id: int, session: Session = Depends(get_session)) -> NewsResponse:
    res = NewsService(session=session).get_news_data_by_id(news_id=news_id)
    session.close()
    return res


@news_router.get("/list/{stock_code}/{page}", response_model=List[NewsListAtt])
def get_news_list_by_name(stock_code: StockCode, page: int, session: Session = Depends(get_session)) -> List[NewsListAtt]:
    news_list = NewsService(session=session).get_news_list_by_stock_code(stock_code=stock_code)
    session.close()
    return news_list
