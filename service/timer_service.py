import datetime
import time

from service.base_service import BaseService

FRIDAY = 0
HOUR = 60
WEEK_TIME = 60 * 24 * 7


class TimerService(BaseService):
    def minute_timer():
        last_time = datetime.datetime.now()
        while True:
            current_time = datetime.datetime.now()
            if current_time - last_time >= datetime.timedelta(minutes=1):
                yield True
                last_time = current_time
            else:
                yield False
            time.sleep(1)

    def training_timer():

        while True:
            current_time = datetime.datetime.now()
            if current_time.weekday() == FRIDAY:
                if current_time.time() == datetime.time(hour=17):
                    yield True
                else:
                    yield False
                time.sleep(HOUR)
            else:
                yield False
            time.sleep(WEEK_TIME)
