from fastapi import APIRouter

from service.stock_service import StockService

stock_router = APIRouter()


@stock_router.get("/current_price/{item_code}")
def get_current_stock_price(item_code: str):
    StockService().get_stock_data_now(item_code=item_code)
