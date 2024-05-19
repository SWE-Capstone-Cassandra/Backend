from fastapi import APIRouter

from service.news_service import NewsService
from service.stock_service import StockService

news_router = APIRouter()


@news_router.get("/{news_id}")
def get_news_by_news_id(news_id: int):
    return NewsService().get_news_data(news_id=news_id)


@news_router.get("/get/news_list/{item_code}/{page}")
def get_news_list_by_code(item_code: str, page: int):
    return NewsService().get_news_list(item_name=item_code, page=page)
