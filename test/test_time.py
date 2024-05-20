from datetime import datetime


time_now = datetime.now().time()
time_now = time_now.strftime("%H%M")
print(time_now)
