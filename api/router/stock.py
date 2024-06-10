import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import get_session
from service.stock_service import StockService

stock_router = APIRouter()


@stock_router.post("/save/{price}")
def save_stock_data(price: int, session: Session = Depends(get_session)):
    res = StockService(session=session).save_stock_data(price=price)
    session.commit()
    return res


@stock_router.get("/get/data_now")
def get_stock_data(session: Session = Depends(get_session)):
    res = StockService(session=session).get_stock_data_now()
