import sys
import concurrent
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import queue
from multiprocessing.connection import Client
def fin_loop(fut_queue, exit_queue):
    client = Client(r'\\.\pipe\detectResult', authkey=b'ctRestruct')
    while True:
        try:
            exit_cmd = exit_queue.get_nowait()
            if exit_cmd:
                break
        except queue.Empty:
            pass
        try:
            fut = fut_queue.get(timeout=1)
        except queue.Empty:
            continue
        print('6666 get fut')
        try:
            filename = fut.result(timeout=1)
        except concurrent.futures._base.TimeoutError:
            continue
        print('7777 get filename', filename)
        client.send(filename)
        client.close()

pool = ThreadPoolExecutor(max_workers=10)
def do_thing():
    import time
    time.sleep(1)
    print('do')
    return time.time()


def snap():
    fut = pool.submit(do_thing)
    print('do')
    return fut



if __name__  == "__main__":
    fut_queue = queue.Queue()
    exit_queue = queue.Queue()
    fin_thread = Thread(target=fin_loop, args=(fut_queue,exit_queue), daemon=True)
    fin_thread.start()
    print('in client')
    while True:
        try:
            cmd = sys.stdin.readline()
            if cmd.startswith('snap'):  
                fut = snap()
                fut_queue.put(fut)
            else:
                break
        except  Exception as e:
            print('!!!!',e)
            raise e
               
