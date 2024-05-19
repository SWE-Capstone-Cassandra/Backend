from fastapi import APIRouter

from api.handler.news import news_router
from api.handler.stock import stock_router

main_router = APIRouter()

main_router.include_router(news_router, prefix="/news", tags=["news"])
main_router.include_router(stock_router, prefix="/stock", tags=["stock"])
