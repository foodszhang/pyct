from thread import ExitableThread
from threading import get_native_id

import time
def sleep_many():
    for i in range(100):
        print("sleep_many", i, get_native_id())
        time.sleep(1)

if __name__ == "__main__":
    thread = ExitableThread(target=sleep_many)
    thread.start()
    time.sleep(3)
    thread.stop()
    thread.join()
    print("main thread exit")
