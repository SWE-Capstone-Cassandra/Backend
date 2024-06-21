import datetime

from config import get_session
from service.model_service import ModelService
from utils.enum.stock_list import StockList

START_DAY = datetime.datetime(year=2024, month=6, day=20, hour=0, minute=0)
END_DAY = datetime.datetime(year=2024, month=6, day=21, hour=0, minute=0)
stock = StockList.CJ_JAIL_JAEDANG

with get_session() as session:

    ModelService(session=session).request_prediction(start_day=START_DAY, end_day=END_DAY, stock=stock)
    session.commit()
