import dearpygui.dearpygui as dpg 
import cv2
import array
import numpy as np
def create_ct_image_window():
    with dpg.window(label='CT图像采集展示',tag='ct-image-window'):
        ar = cv2.imread('dark.tif')
        # 标准化
        h,w = ar.shape[:2]
        cv2.normalize(ar, ar, 0, 255, cv2.NORM_MINMAX)

        #等比例调整大小
        width = 500
        ar = cv2.resize(ar, (width, int(width*h/w)))

        # 转为单行
        ar = ar.ravel()
        ar = np.asfarray(ar, dtype='f')
        ar = np.true_divide(ar, 255.0)
        with dpg.texture_registry(show=False):
            texture_id = dpg.add_raw_texture(width=width, height=int(width*h/w), default_value=ar, tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        dpg.add_image(texture_id)
