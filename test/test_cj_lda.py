import datetime

import pandas as pd

if __name__ == "__main__":
    from ai_model.data_controller import DataController

    news_dataset = pd.read_csv(filepath_or_buffer="/home/tako4/capstone/backend/Backend/data/cj_230701-240630.csv")
    news_dataset = news_dataset[news_dataset["date"] != "date"]
    try:
        news_dataset["date_time"] = pd.to_datetime(news_dataset["date"])

    except ValueError as e:
        print(f"Error: {e}")

    # 다른 CSV 파일에서 데이터프레임 로드
    stock_dataset = pd.read_excel("/home/tako4/capstone/backend/Backend/data/A097950_min_chart_20240611_20220527.xlsx")

    # # date와 time을 결합하여 datetime 객체로 변환
    try:
        stock_dataset["date_time"] = stock_dataset.apply(
            lambda row: datetime.datetime.strptime(str(row["Date"]) + " " + str(row["Time"]), "%Y%m%d %H%M%S"), axis=1
        )
    except ValueError as e:
        print(f"Error: {e}")

    try:
        DataController().train_news_dataset(news_dataset=news_dataset, stock_dataset=stock_dataset)
        print("done")
    except Exception as ex:
        print("무슨 에러지?", ex)
