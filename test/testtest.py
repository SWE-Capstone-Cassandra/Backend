from datetime import datetime

import pandas as pd

# 예제 데이터 생성
data = {
    "datetime": [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 0, 5), datetime(2023, 1, 1, 0, 10), datetime(2023, 1, 1, 0, 15)],
    "min1": [[10, 12], [20, 12], 30, 40],
    "min5": [15, 25, 35, 45],
    "min15": [20, 30, 40, 50],
    "original": [5, 10, 15, 20],
}

# 데이터프레임 생성
df = pd.DataFrame(data)
df.set_index("datetime", inplace=True)

print("Initial DataFrame:")
print(df)


# 특정 datetime과 열로 값을 검색하여 배열로 반환하는 함수
def search_by_datetime_and_column(dataframe, dt, column):
    try:
        value = dataframe.at[dt, column]
        return [value]
    except KeyError:
        return []


# 검색 예시
result = search_by_datetime_and_column(df, datetime(2023, 1, 1, 0, 5), "min5")
print(f"Values for (2023-01-01 00:05, 'min5'): {result}")

result = search_by_datetime_and_column(df, datetime(2023, 1, 1, 0, 10), "original")
print(f"Values for (2023-01-01 00:10, 'original'): {result}")
