from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config import get_session
from service.stock_service import StockService

stock_router = APIRouter()


@stock_router.post("/save/{time}/{price}")
def save_stock_data(time: int, price: int, session: Session = Depends(get_session)):
    res = StockService(session=session).save_stock_data(time=time, price=price)
    session.commit()
    return res
