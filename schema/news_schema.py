from typing import Optional

from pydantic import BaseModel


class NewsListAtt(BaseModel):
    title: Optional[str] = None
    news_id: Optional[int] = None


class NewsSchema(BaseModel):
    # id: int
    news_id: Optional[int] = None
    date_time: str
    title: Optional[str] = None
    writer: Optional[str] = None
    content: str


class NewsResponse(NewsSchema):
    pass
