from pytimekr import pytimekr
from datetime import datetime, timedelta
import pandas as pd


def adjust_time(time):
    """
    다음 거래일 09:00 설정을 위한 함수
    """

    def next_business_day_9am(current_day):
        next_day = current_day.date() + timedelta(days=1)
        # 다음 날이 공휴일이거나 주말인 경우 다음 가능한 평일로 이동
        while next_day in pytimekr.holidays(next_day.year) or next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return datetime.combine(next_day, datetime.strptime("09:00", "%H:%M").time())

    # 당일 09:00 설정을 위한 함수
    def same_day_9am(current_day):
        # 해당 일이 공휴일이거나 주말인 경우 다음 거래일로 조정
        if current_day.date() in pytimekr.holidays(current_day.year) or current_day.weekday() >= 5:
            return next_business_day_9am(current_day)
        else:
            return datetime.combine(current_day.date(), datetime.strptime("09:00", "%H:%M").time())

    # 금요일 15:30 이후
    if time.weekday() == 4 and time.time() > datetime.strptime("15:30", "%H:%M").time():
        return next_business_day_9am(time + timedelta(days=3))  # 다음 주 월요일 09:00

    # 토요일, 일요일 전체
    elif time.weekday() >= 5:
        return next_business_day_9am(time)

    # 평일 09:00 이전
    elif time.time() < datetime.strptime("09:00", "%H:%M").time():
        return same_day_9am(time)

    # 평일 15:30 이후
    elif time.time() > datetime.strptime("15:30", "%H:%M").time():
        return next_business_day_9am(time)

    # 해당 일이 공휴일인 경우 다음 거래일 09:00으로 조정
    elif time.date() in pytimekr.holidays(time.year):
        return next_business_day_9am(time)

    # 그 외 시간은 현재 시간 유지
    else:
        return time


def calculate_price_change(current_time, minutes, stock_dataset: pd.DataFrame):
    """
    주가 변화량 계산 함수

    Args:
        current_time: DataFrame 객체의 date_time 열 값
        minutes: 특정 분 후의 시간
        stock_dataset: 종목의 1분봉 주가 데이터 세트

    Returns:
        주가 변화량 계산 값
    """

    future_time = current_time + timedelta(minutes=minutes)
    if future_time.time() > datetime.strptime("15:20", "%H:%M").time():
        future_time = (future_time + timedelta(days=1)).replace(hour=9, minute=0)

    current_prices = stock_dataset.loc[stock_dataset["date_time"] == current_time.strftime("%Y-%m-%d %H:%M"), "price"]
    future_prices = stock_dataset.loc[stock_dataset["date_time"] == future_time.strftime("%Y-%m-%d %H:%M"), "price"]

    # print("current_price: ", current_prices)
    # print("future_price: ", future_prices)

    if not current_prices.empty and not future_prices.empty:
        current_price = current_prices.values[0]
        future_price = future_prices.values[0]
        return (future_price - current_price) / current_price * 100
    else:
        return None  # 데이터가 없는 경우 None 반환
