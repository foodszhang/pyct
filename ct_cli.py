from serial_controller import ZolixMcController
import yaml
import os
import subprocess
import typer
import pipe
from typing import Optional, Annotated
from threading import Thread
import time
import cv2
import numpy as np
import sys
if __name__ == "__main__":
    py34 = os.environ.get("py34", None)
    ZC = ZolixMcController('COM4', 19200)
    #最高速5000 96s
    ZC.set_speed(5)
    ZC.set_init_speed(5)
    #ZC.set_stage_rate(180)
    #ZC.query()
    last_t = time.time()
    if not py34:
        typer.echo("py34环境变量未配置!请检查系统环境变量, 保证py34环境变量指向3.4版本python.exe的路径")
        sys.exit(1)
    sub = subprocess.Popen([py34, "detector.py", "seq","100", '100', "60"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert sub.stdout
    assert sub.stdin
    ready_cmd=sub.stdout.readline()
    if not ready_cmd.startswith(b'READY'):
        print("ready_cmd333", ready_cmd)
        sys.exit(1)
    sub.stdin.write("start\n".encode())
    sub.stdin.flush()
    ZC.motion_rotation(70)
    cmd = sub.stdout.readline()
    print('!!!!', cmd)

