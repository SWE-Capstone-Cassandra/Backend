from pydantic import BaseModel


class CreateNewsPrediction(BaseModel):
    news_id: str
    time: str


class NewsPredictionSchema(BaseModel):
    min_1: float
    min_5: float
    min_15: float
    hour_1: float
    day_1: float
    stock_price: float


class NewsPredictionResponse(NewsPredictionSchema):
    pass
