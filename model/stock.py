from datetime import datetime

from sqlalchemy import DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from config import Base


class Stock(Base):
    __tablename__ = "samsung"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
