from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from config import Base, engine


class NewsPrediction(Base):
    __tablename__ = "NewsPrediction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    news_id: Mapped[str] = mapped_column(String)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    min_1: Mapped[int] = mapped_column(Integer)
    min_5: Mapped[int] = mapped_column(Integer)
    min_15: Mapped[int] = mapped_column(Integer)
    min_60: Mapped[int] = mapped_column(Integer)
    day_1: Mapped[int] = mapped_column(Integer)


Base.metadata.create_all(bind=engine)
