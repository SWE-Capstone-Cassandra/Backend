from datetime import datetime

from sqlalchemy import DateTime, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from config import Base


class NewsPrediction(Base):
    __tablename__ = "news_prediction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    news_id: Mapped[int] = mapped_column(Integer)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    min_1: Mapped[float] = mapped_column(Float)
    min_5: Mapped[float] = mapped_column(Float)
    min_15: Mapped[float] = mapped_column(Float)
    hour_1: Mapped[float] = mapped_column(Float)
    day_1: Mapped[float] = mapped_column(Float)
