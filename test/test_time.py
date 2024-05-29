import os
import sys
from datetime import datetime

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
time_now = datetime.now().time()
time_now = time_now.strftime("%H%M")
print(time_now)
