import dearpygui.dearpygui as dpg
from gui.helper import show_info 
from serial_controller import ZolixMcController
import yaml
import os
import subprocess
import pipe
from threading import Thread
import time
import cv2
import numpy as np
from gui.logger import logger
Config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

def scan_start_callback(sender, app_data, user_data):
    server_thread = Thread(
        target=pipe.detector_server,
        args=(r"\\.\pipe\detectResult", b"ctRestruct", snap_one_client),
        daemon=True,
    )
    server_thread.start()
    py34 = os.environ.get("py34", '')
    if not py34:
        show_info("Warning", "py34环境变量未配置!请检查系统环境变量, 保证py34环境变量指向3.4版本python.exe的路径")
        return
    sub = subprocess.Popen([py34, "detector.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    #logger.log_info("开启探测器进程")
    config = Config.get("ZolixMcController", None)
    if not config:
        show_info("Warning", "转台控制器配置出错!请检查config.yaml文件")
        return
    controller = ZolixMcController(config["port"], config["baudrate"])
    speed = dpg.get_value("motion-speed-text") or config["speed"]
    controller.set_speed(speed)
    degree = dpg.get_value("motion-degree-text") or config["degree"]
    filepath = dpg.get_value("ct-image-path-text")
    set_progress_bar("ct-scan-progress", 10)
    assert sub.stdin
    for i in range(round(360 / degree)):
        #logger.log_info(f"旋转采集:采集第{i}张图片")
        full_filename = os.path.join(filepath, f'{i}.tif')
        cmd_str = "snap {}\n".format(full_filename)
        sub.stdin.write(cmd_str.encode())
        sub.stdin.flush()
        result = sub.stdout.readline()
        if result != b'ok\r\n':
            print("角度{}采集失败".format(i))
            print('!!!!!result', result)
            #logger.log_info("旋转采集:采集失败")
            break
        controller.motion_rotation(degree)
        set_progress_bar("ct-scan-progress", 10 + 90 * i / round(360 / degree))

    sub.stdin.write("exit\n".encode())
    sub.stdin.flush()
    server_thread.join()
    set_progress_bar("ct-scan-progress", 100)
    #logger.log_info("旋转采集:采集结束")
    


def scan_end_callback(sender, app_data, user_data):
    pass


def create_ct_image_file_dialog():
    with dpg.file_dialog(
        directory_selector=True,
        show=False,
        tag="ct-file-dialog",
        width=800,
        callback=ct_image_file_callback,
    ):
        dpg.add_text("选择图片存储目录")


def ct_image_file_callback(sender, app_data, user_data):
    file_path = None
    file_path = app_data["file_path_name"]
    dpg.set_value("ct-image-path-text", file_path)


def snap_one_client(conn):
    try:
        while True:
            filename, buf = conn.recv()
            split_name = filename.split(".")
            #logger.log_info(f"recv filename: {filename}")
            w = 1944
            h = 1536
            ar = np.frombuffer(buf, dtype=np.uint16).reshape(w, h)
            if len(split_name) > 1:
                cv2.imwrite(f'{".".join(split_name[:-1])}.tif', ar)
            else:
                cv2.imwrite(f"{filename}.tif", ar)

            ar = cv2.normalize(ar, None, 0, 255, cv2.NORM_MINMAX)
            width = 500
            ar = cv2.resize(ar, (width, int(width*h/w)))
            ar = cv2.cvtColor(ar,cv2.COLOR_GRAY2BGR)
            ar = ar.ravel()
            ar = np.asfarray(ar, dtype='f')
            ar = np.true_divide(ar, 255.0)
            dpg.set_value("texture_tag", ar)
    except EOFError:
        #logger.log_info("连接断开， 采集结束")
        print("close!")


def snap_callback(sender, app_data, user_data):
    server_thread = Thread(
        target=pipe.detector_server,
        args=(r"\\.\pipe\detectResult", b"ctRestruct", snap_one_client),
        daemon=True,
    )
    server_thread.start()
    py34 = os.environ.get("py34", '')
    if not py34:
        show_info("Warning", "py34环境变量未配置!请检查系统环境变量, 保证py34环境变量指向3.4版本python.exe的路径")
        return

    sub = subprocess.Popen([py34, "detector.py"], stdin=subprocess.PIPE)
    #logger.log_info("单张采集:开启探测器进程")
    set_progress_bar("ct-snap-progress", 10)
    filename = dpg.get_value("ct-snap-filename-text")
    full_filename = os.path.join(dpg.get_value("ct-image-path-text"), filename)
    cmd_str = "snap {}\n".format(full_filename)
    assert sub.stdin is not None
    sub.stdin.write(cmd_str.encode())
    sub.stdin.flush()
    set_progress_bar("ct-snap-progress", 50)
    sub.stdin.write("exit\n".encode())
    sub.stdin.flush()
    server_thread.join()
    set_progress_bar("ct-snap-progress", 100)
    #logger.log_info("单张采集:采集结束")


def set_progress_bar(tag, value):
    dpg.configure_item(tag, overlay=f"{value}%")
    dpg.set_value(tag, value)

def create_ct_control_window():
    create_ct_image_file_dialog()
    with dpg.window(label="CT采集控制", tag="ct-control-window"):
        snap_state = False
        with dpg.child_window(height=100):
            dpg.add_text("图片存储路径")
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    default_value=r"E:\for_new_test\data\alway", readonly=True, tag="ct-image-path-text"
                )
                dpg.add_button(
                    label="浏览", callback=lambda: dpg.show_item("ct-file-dialog")
                )
        with dpg.child_window(height=120):
            with dpg.group(horizontal=True):
                dpg.add_text("采集文件名")
                dpg.add_input_text(
                    default_value="dark.tif", tag="ct-snap-filename-text"
                )
            dpg.add_button(label="单张采集", callback=snap_callback)
            with dpg.group(horizontal=True):
                dpg.add_text("采集进度")
                dpg.add_progress_bar(
                    default_value=0, overlay="0%", tag="ct-snap-progress"
                )
        with dpg.child_window(height=200):
            with dpg.group(horizontal=True):
                dpg.add_text("转台速度")
                dpg.add_input_int(default_value=2000, tag="motion-speed-text")
            with dpg.group(horizontal=True):
                dpg.add_text("每次旋转角度")
                dpg.add_input_float(default_value=1, tag="motion-degree-text")
            dpg.add_button(label="开始采集", callback=scan_start_callback)
            with dpg.group(horizontal=True):
                dpg.add_text("采集进度")
                dpg.add_progress_bar(
                    default_value=0, overlay="0%", tag="ct-scan-progress"
            )

