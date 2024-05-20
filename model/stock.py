from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column
from config import Base


class Stock(Base):
    __tablename__ = "samsung"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[int] = mapped_column(Integer)
    time: Mapped[int] = mapped_column(Integer)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
