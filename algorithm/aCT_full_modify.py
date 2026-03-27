import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, simpledialog
from PIL import Image
import gc
import os

def badPixels(dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down, Map_I):
    x0, y0 = np.where(dark_ave_nor > dark_up)
    x1, y1 = np.where(dark_ave_nor < dark_down)
    x2, y2 = np.where(bright_ave_nor > bright_up)
    x3, y3 = np.where(bright_ave_nor < bright_down)
    
    # 从以上的四组数据中确定补偿坏线之后的坏点，并剔除重复点，按升序排列
    axis_index = np.unique(np.vstack((np.column_stack((x0, y0)), np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3)))), axis=0)
    FILTER_LENGTH = 10
    x4 = np.where(axis_index[:, 0] < FILTER_LENGTH)[0]
    x5 = np.where(axis_index[:, 0] > (1944 - FILTER_LENGTH))[0]
    del_index = np.concatenate((x4, x5))
    axis_index = np.delete(axis_index, del_index, axis=0)
    
    # 零星点
    for i in range(len(axis_index)):
        x, y = axis_index[i]
        filter = Map_I[x - FILTER_LENGTH:x + FILTER_LENGTH + 1, y - FILTER_LENGTH:y + FILTER_LENGTH + 1]
        add_value = np.mean(filter)
        Map_I[x, y] = add_value
    
    Map_D = Map_I
    return Map_D
    


def define():
    # 参数设置
    Binning_flag = 1
    w = 1944 #30
    h = 1536 #30

    # 读入空扫和暗电流数据
    # 待检查数据文件
    root = Tk()
    root.withdraw()  # 隐藏Tkinter窗口

    # 打开文件对话框，等待用户选择文件
    filename_dark = filedialog.askopenfilename(title='Select the Dark-file')

    # 检查用户是否选择了文件
    if filename_dark:
        dark_ave = np.flipud(np.fromfile(filename_dark, dtype=np.uint16)[:w*h].reshape((w, h)))
        # dark_ave = np.flipud(np.fromfile(filename_dark, dtype=np.uint16)[:w * h])
        # 在这里可以继续处理 dark_ave
    else:
        print("用户没有选择文件。")

    root.destroy()  # 关闭Tkinter窗口

    root = Tk()
    root.withdraw()  # 隐藏Tkinter窗口

    # 打开文件对话框，等待用户选择文件
    filename_bright = filedialog.askopenfilename(title='Select the Bright-file')

    # 检查用户是否选择了文件
    if filename_bright:
        bright_ave = np.flipud(np.fromfile(filename_bright, dtype=np.uint16)[:w * h].reshape((w, h)))
    else:
        print("用户没有选择文件。")

    root.destroy()  # 关闭Tkinter窗口

    dark_ave[1744:, :] = np.mean(np.mean(dark_ave))
    bright_ave[1744:, :] = np.mean(np.mean(bright_ave))
    # dark_ave[20:, :] = np.mean(np.mean(dark_ave))
    # bright_ave[20:, :] = np.mean(np.mean(bright_ave))

    # 像素拼合
    if Binning_flag == 2:
        dark_ave = bine(dark_ave)
        bright_ave = bine(bright_ave)

    if Binning_flag == 4:
        dark_ave = bine(dark_ave)
        bright_ave = bine(bright_ave)
        dark_ave = bine(dark_ave)
        bright_ave = bine(bright_ave)

    # 直方图投影
    bright_ave_nor = bright_ave / np.max(bright_ave)
    dark_ave_nor = dark_ave / np.max(dark_ave)
    size_image = bright_ave_nor.shape

    # 进行垂直灰度投影
    V = np.sum(bright_ave_nor, axis=0) / len(bright_ave_nor)

    # 进行水平灰度投影
    L = np.sum(bright_ave_nor, axis=1) / len(bright_ave_nor)

    # 数据显示，以确定门限
    plt.figure()
    plt.plot(bright_ave_nor)
    plt.grid(True)
    plt.title('bright data (通过该图确定坏点阈值门限)')

    plt.figure()
    plt.plot(dark_ave_nor)
    plt.grid(True)
    plt.title('dark data (通过该图确定坏点阈值门限)')

    plt.figure()
    plt.plot(L)
    plt.grid(True)
    plt.title('bright data (通过该图确定坏线阈值门限-第1维度)')

    plt.figure()
    plt.plot(V)
    plt.grid(True)
    plt.title('bright data (通过该图确定坏线阈值门限-第2维度)')

    # 输入确定门限
    # 这里需要从上面四张图片看出来，bright picture's upper threshold and bottom threshold,
    # dark picture's upper threshold and bottom threshold
    # 最后设置坏线阈值门限-line_1, 设置坏线阈值门限-line_2
    bright_up = float(simpledialog.askstring("Input", "请设置坏点阈值门限-bright_up"))
    bright_down = float(simpledialog.askstring("Input", "请设置坏点阈值门限-bright_down"))
    dark_up = float(simpledialog.askstring("Input", "请设置坏点阈值门限-dark_up"))
    dark_down = float(simpledialog.askstring("Input", "请设置坏点阈值门限-dark_down"))
    line_1 = float(simpledialog.askstring("Input", "请设置坏线阈值门限-line_1"))
    line_2 = float(simpledialog.askstring("Input", "请设置坏线阈值门限-line_2"))

    return w, h, dark_ave, bright_ave, bright_up, bright_down, dark_up, dark_down, line_1, line_2, bright_ave_nor,dark_ave_nor, L, V

def bine(array):
    # 实现像素拼合的函数
    pass


# histogramcreate 函数被拆分成3个，不用for循环写，而是用numpy的内置函数
def find_bad_line(V, line_1):
    # bad line indice, row or line
    index_row = np.where(V < line_1)[0]
    return index_row


def find_bad_line_intervals(index_row):
    # find out the bad line intervals
    if len(index_row) == 0:
        return np.array([])

    diff = np.diff(index_row)
    index_set_row = np.split(index_row, np.where(diff != 1)[0] + 1)
    index_set_row = np.array([(interval[0], interval[-1]) for interval in index_set_row])
    return index_set_row


def calculate_bad_line_width(index_set_row, index_set_col):
    width_row = np.diff(index_set_row[::2])
    width_col = np.diff(index_set_col[::2])

    return width_row, width_col


def histogramcreate(line_1, line_2, V, L):
    index_row = find_bad_line(V, line_1)
    index_col = find_bad_line(L, line_2)

    index_set_row = find_bad_line_intervals(index_row)
    index_set_col = find_bad_line_intervals(index_col)

    width_row, width_col = calculate_bad_line_width(index_set_row, index_set_col)

    return index_col, index_row, width_col, width_row, index_set_row, index_set_col, L, V


def read_file(index_set_col, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down):
    # 返回CT图像的集合(w, h, s)
    # 读取和保存文件的路径，记得这是到文件夹
    Path_read = filedialog.askdirectory(title='Select the Read-Path')
    Path_Save = filedialog.askdirectory(title='Select the Save-Path')
    w = 1536
    h = 1944

    # 注意文件格式，这里打开的是tif文件
    fileNames = [filename for filename in os.listdir(Path_read) if filename.endswith('.tif')]

    pic_set = np.zeros((h, w, 1), dtype=np.uint16)
    num_files = len(fileNames)

    for index, filename in enumerate(fileNames):
        progress = (index + 1) / num_files
        print(f'Computing {progress * 100:.2f}%')

        with open(os.path.join(Path_read, filename), 'rb') as fid:
            pic = np.fromfile(fid, dtype=np.uint16)[0:w * h].reshape((w, h))
            fid.seek(1, os.SEEK_CUR)

            if index_set_row.any():
                pic = supplementlinerow(pic, index_set_row, 'r')
            #if index_set_col.any():
            #    pic = supplementlinerow(pic, index_set_col, 'c')

            pic = np.flipud(pic.T)
            pic = badPixels(dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down, pic)
            plt.imshow(pic / np.max(pic), cmap='gray')
            plt.axis('off')
            writefile(Path_Save, pic, [filename])
            plt.imsave(os.path.join(Path_Save, f'{index + 1}.png'), pic, format='png', cmap='gray')
            plt.close()

    return pic_set, num_files


def supplementlinerow(image, set_image, sign):
    # 补偿坏线
    if sign == 'r': # row
        for row_range in set_image:
            start, end = row_range[0], row_range[1] + 1
            line = (image[:, start - 1] + image[:, end]) / 2
            image[:, start:end] = np.tile(line[:, np.newaxis], (1, end - start))
    elif sign == 'c': # col
        for col_range in set_image:
            start, end = col_range[0], col_range[1] + 1
            line = (image[start - 1, :] + image[end, :]) / 2
            image[start:end, :] = np.tile(line[np.newaxis, :], (end - start, 1))

    return image


def writefile(Path, CT_set, filename):
    # 写入CT图像数据到文件
    full_path = os.path.join(Path, filename[0])
    with open(full_path, 'wb') as fid:
        fid.write(CT_set.tobytes())



if __name__ == "__main__":
    # 清除所有变量
    gc.collect()
    
    # 调用 define 函数获取变量值
    w, h, dark_ave, bright_ave, bright_up, bright_down, dark_up, dark_down, line_1, line_2, bright_ave_nor, dark_ave_nor, L, V = define()
    
    # 调用 histogramcreate 函数获取变量值
    index_col, index_row, width_col, width_row, index_set_row, index_set_col, L, V = histogramcreate(line_1, line_2, V, L)
    
    # 调用 read_file 函数获取变量值
    pic_set, len_set = read_file(index_set_col, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down)
    
    # 关闭所有图形窗口
    plt.close('all')
