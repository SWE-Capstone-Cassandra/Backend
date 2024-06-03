from fastapi import APIRouter

from service.model_service import ModelService


model_router = APIRouter()


@model_router.get("/prediction/{news_id}")
def get_prediction_by_news_id(news_id: int):
    return ModelService().get_prediction_by_news_id(news_id=news_id)
