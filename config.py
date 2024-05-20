from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, DeclarativeBase


user = "cap"
password = "capstone"
sever = "localhost:5432"
database = "capstone"

DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{sever}/{database}"


class Base(DeclarativeBase):
    def __repr__(self) -> str:
        return str({c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs})


def get_engine():
    engine = create_engine(DATABASE_URL)
    return engine


def get_session():
    engine = get_engine()
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session


def create_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
