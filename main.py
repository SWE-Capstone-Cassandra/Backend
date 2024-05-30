import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router.model import model_router
from api.router.news import news_router
from api.router.stock import stock_router
from config import create_db

app = FastAPI()

app.include_router(stock_router, prefix="/stock")
app.include_router(news_router, prefix="/news")
app.include_router(model_router, prefix="/model")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    create_db()

    uvicorn.run("main:app", host="0.0.0.0", port=8096, workers=3)
