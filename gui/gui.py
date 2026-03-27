import dearpygui.dearpygui as dpg
import os
import shutil
from gui.ct_control import create_ct_control_window
from gui.ct_image import create_ct_image_window
from gui.logger import create_logger


def set_font():
    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        default_font = dpg.add_font("msyh.ttc", 20)
        with dpg.font("msyh.ttc", 20) as font1:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
    dpg.bind_font(dpg.last_container())


def change_active_window(sender, app_data, user_data):
    if dpg.get_value(sender) == True:
        dpg.show_item(user_data)
    else:
        dpg.hide_item(user_data)


def create_viewport_menu_bar():
    with dpg.viewport_menu_bar():
        with dpg.menu(label="项目"):
            dpg.add_menu_item(label="保存项目")
            dpg.add_menu_item(label="首选项")
        with dpg.menu(label="窗口"):
            dpg.add_checkbox(
                label="CT采集设置窗口",
                default_value=True,
                callback=change_active_window,
                user_data="CT-contorl-window",
            )
            dpg.add_checkbox(
                label="CT结果展示窗口",
                default_value=True,
                callback=change_active_window,
                user_data="ct-image-window",
            )
            dpg.add_button(
                label="保存布局",
                callback=lambda: dpg.save_init_file("user_custom_layout.ini"),
            )


def start_gui():
    dpg.create_context()
    dpg.configure_app(
        docking=True, docking_space=True, init_file="user_custom_layout.ini"
    )  # must be called before create_viewport
    set_font()
    dpg.create_viewport(title="Reconstruct ToolBox", width=1240, height=800)
    if not os.path.exists("user_custom_layout.ini"):
        shutil.copy("custom_layout.ini", "user_custom_layout.ini")
    # with dpg.window(label='预设流程',tag='toolbar'):
    #    dpg.add_button(label='CT采集')
    with dpg.theme(tag="button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (74, 45, 115))
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive, (75, 54, 105)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered, (107, 81, 143)
            )
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2 * 5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2 * 3, 2 * 3)
    dpg.bind_theme('button_theme')

    create_ct_control_window()
    create_viewport_menu_bar()
    create_ct_image_window()
    create_logger()


    dpg.setup_dearpygui()
    dpg.show_viewport()
    # dpg.set_primary_window("main", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
