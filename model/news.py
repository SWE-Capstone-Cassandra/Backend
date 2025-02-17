from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from config import Base
from utils.enum.stock_code import StockCode


class News(Base):
    __tablename__ = "news"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_code: Mapped[StockCode] = mapped_column(String)
    news_id: Mapped[int] = mapped_column(Integer, nullable=True)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    title: Mapped[str] = mapped_column(String, nullable=True)
    writer: Mapped[str] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(String)
