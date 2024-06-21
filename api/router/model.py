from fastapi import APIRouter, Depends

from config import get_session
from schema.news_prediction import NewsPredictionResponse
from service.model_service import ModelService

model_router = APIRouter()


@model_router.get("/prediction/{news_id}")
def get_prediction_by_news_id(news_id: str, session=Depends(get_session)) -> NewsPredictionResponse:
    news_id = int(news_id)
    return ModelService(session=session).get_prediction_by_news_id(news_id=news_id)
