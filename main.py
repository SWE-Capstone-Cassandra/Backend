from fastapi import FastAPI

app = FastAPI()

app.add_
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 프로그램 요구사항
# 인공지능이랑 연결되어이써야하한다.
# 오늘 주가 데이터들을 전부 가지고 있어야 한다.
# 오늘 뉴스 데이터를 가지고 있어야 한다.
# 오늘 뉴스 데이터와 주가 데이터를 통합해서 보여줘야한다.
