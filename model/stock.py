from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from config import Base
from utils.enum.stock_code import StockCode


class Stock(Base):
    __tablename__ = "stock"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_code: Mapped[StockCode] = mapped_column(String)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
