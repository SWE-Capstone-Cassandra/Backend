import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import get_session
from service.stock_service import StockService
from utils.enum.stock_code import StockCode

stock_router = APIRouter()


@stock_router.post("/save/{stock_code}/{price}")
def save_stock_data(stock_code: StockCode, price: int, session: Session = Depends(get_session)):
    res = StockService(session=session).save_stock_data(stock_code=stock_code, price=price)
    session.commit()
    return res


@stock_router.get("/get/data_now/{stock_code}")
def get_stock_data(stock_code: StockCode, session: Session = Depends(get_session)):
    res = StockService(session=session).get_stock_data_now(stock_code=stock_code)
    return res
