from dearpygui_ext.logger import mvLogger
import dearpygui.dearpygui as dpg
logger = None
def create_logger():
    global logger
    if logger:
        return
    log = mvLogger()
    dpg.set_item_label(log.window_id, "日志")
    logger = log
