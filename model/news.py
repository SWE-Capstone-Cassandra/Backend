from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from config import Base


class News(Base):
    __tablename__ = "news"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_url: Mapped[int] = mapped_column(String, nullable=True)
    date: Mapped[int] = mapped_column(Integer, index=True, nullable=True)
    time: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String, nullable=True)
    writer: Mapped[str] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(String)
