from fastapi import APIRouter

from schema.news_prediction import NewsPredictionResponse
from service.news_service import NewsService

news_router = APIRouter()


@news_router.get("/{news_id}")
def get_news_by_news_id(news_id: int):
    return NewsService().get_news_data(news_id=news_id)


@news_router.get("/prediction/{item_code}/{news_id}", response_model=NewsPredictionResponse)
def get_prediction_by_news_id(item_code: str, news_id: int) -> NewsPredictionResponse:
    return NewsService().get_prediction(item_code=item_code, news_id=news_id)


@news_router.get("/list/{item_code}/{page}")
def get_news_list_by_code(item_code: str, page: int):
    return NewsService().get_news_list(item_name=item_code, page=page)
