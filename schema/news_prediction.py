from pydantic import BaseModel


class CreateNewsPrediction(BaseModel):
    news_id: str
    time: str


class NewsPrediction(CreateNewsPrediction):
    min_1: int
    min_5: int
    min_15: int
    min_60: int
    day_1: int


class NewsPredictionResponse(NewsPrediction):
    pass
