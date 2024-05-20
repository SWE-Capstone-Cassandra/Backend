from fastapi import APIRouter


from service.news_service import NewsService
from service.stock_service import StockService

stock_router = APIRouter()


@stock_router.post("/save/{time}/{price}")
def save_stock_data(time: int, price: int):
    return StockService().save_stock_data(time=time, price=price)
