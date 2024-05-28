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


# 프로그램 요구사항
# 인공지능이랑 연결되어이써야하한다.
# 오늘 주가 데이터들을 전부 가지고 있어야 한다.
# 오늘 뉴스 데이터를 가지고 있어야 한다.
# 오늘 뉴스 데이터와 주가 데이터를 통합해서 보여줘야한다.
