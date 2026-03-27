import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, simpledialog
from PIL import Image
import gc
import os
import random
import multiprocessing
import time
import concurrent.futures


def badPixels(dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down, pic):
    x0, y0 = np.where(dark_ave_nor > dark_up)
    x1, y1 = np.where(dark_ave_nor < dark_down)
    x2, y2 = np.where(bright_ave_nor > bright_up)
    x3, y3 = np.where(bright_ave_nor < bright_down)

    # 从以上的四组数据中确定补偿坏线之后的坏点，并剔除重复点，按升序排列
    axis_index = np.unique(np.vstack(
        (np.column_stack((x0, y0)), np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3)))),
        axis=0)
    FILTER_LENGTH = 10
    x4 = np.where(axis_index[:, 0] < FILTER_LENGTH)[0]
    x5 = np.where(axis_index[:, 0] > (1944 - FILTER_LENGTH))[0]
    del_index = np.concatenate((x4, x5))
    axis_index = np.delete(axis_index, del_index, axis=0)

    # 零星点
    for i in range(len(axis_index)):
        x, y = axis_index[i]
        if FILTER_LENGTH <= x < (pic.shape[0] - FILTER_LENGTH) and FILTER_LENGTH <= y < (pic.shape[1] - FILTER_LENGTH):
            filter_py = pic[x - FILTER_LENGTH:x + FILTER_LENGTH + 1, y - FILTER_LENGTH:y + FILTER_LENGTH + 1]
            add_value = np.mean(filter_py)
            pic[x, y] = add_value

    return pic


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


def process_file_one(fileNames, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down,
                     directory_in, directory_out, w=1536, h=1944):
    num_files = len(fileNames)

    for index, filename in enumerate(fileNames):
        progress = (index + 1) / num_files
        print(f'Computing {progress * 100:.2f}%')

        with open(os.path.join(directory_in, filename), 'rb') as fid:
            pic = np.fromfile(fid, dtype=np.uint16)[0:w * h].reshape((w, h))
            fid.seek(1, os.SEEK_CUR)

            if index_set_row.any():
                pic = supplementlinerow(pic, index_set_row, 'r')
                # if index_set_col.any():
                #    pic = supplementlinerow(pic, index_set_col, 'c')

                pic = np.flipud(pic.T)
                pic = badPixels(dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down, pic)
            # plt.imshow(pic / np.max(pic), cmap='gray')
            plt.axis('off')
            writefile(directory_out, pic, filename)
            plt.imsave(os.path.join(directory_out, f'{index + 1}.png'), pic, format='png', cmap='gray')
            plt.close()


def process_file(filename, index, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up,
                 bright_down, directory_in, directory_out, w=1536, h=1944, progress_queue=None):
    file_read = os.path.join(directory_in, filename)
    k = 0
    with open(file_read, 'rb') as fid:
        pic = np.fromfile(fid, dtype=np.uint16)[0:w * h].reshape((w, h))
        fid.seek(1, os.SEEK_CUR)

        if index_set_row.any():
            pic = supplementlinerow(pic, index_set_row, 'r')
        # if index_set_col.any():
        #    pic = supplementlinerow(pic, index_set_col, 'c')

        pic = np.flipud(pic.T)
        pic = badPixels(dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down, pic)
        plt.axis('off')
        writefile(directory_out, pic, filename)
        plt.imsave(os.path.join(directory_out, f'{index + 1}.png'), pic, format='png', cmap='gray')
        plt.close()

        if progress_queue:
            k += 1
            progress_queue.put(k)


def read_file(index_set_col, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up, bright_down,
              directory_in, directory_out, w=1536, h=1944):
    # 返回CT图像的集合(w, h, s)

    # 注意文件格式，这里打开的是tif文件
    fileNames = [filename for filename in os.listdir(directory_in) if filename.endswith('.tif')]

    pic_set = np.zeros((h, w, 1), dtype=np.uint16)

    thread_count = multiprocessing.cpu_count()  # 获取当前CPU的线程数

    progress_queue = multiprocessing.Queue()  # 创建一个队列用于进程间通信
    k = 0
    if thread_count >= 2:  # 当线程数量大于1时并行处理
        processes = []
        for index, filename in enumerate(fileNames):
            p = multiprocessing.Process(target=process_file, args=(
                filename, index, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up,
                bright_down, directory_in, directory_out, w, h, progress_queue))
            processes.append(p)
            p.start()

        # 实时输出处理进度

        while any(p.is_alive() for p in processes):
            while not progress_queue.empty():
                progress = progress_queue.get()
                k += float(progress)
                print(f'Processing: {k / len(fileNames) * 100:.2f}% completed', end='\r', flush=True)

    elif thread_count == 1:  # 当线程数量等于1时线性处理
        process_file_one(fileNames, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up,
                         bright_down, directory_in, directory_out, w=1536, h=1944)
    else:
        print("There is no cpu in your computer.")


def supplementlinerow(image, set_image, sign):
    # 补偿坏线
    if sign == 'r':  # row
        for row_range in set_image:
            start, end = row_range[0], row_range[1] + 1
            if start - 1 >= 0 and end <= image.shape[1]:
                line = (image[:, start - 1] + image[:, end]) / 2
                image[:, start:end] = np.tile(line[:, np.newaxis], (1, end - start))
    elif sign == 'c':  # col
        for col_range in set_image:
            start, end = col_range[0], col_range[1] + 1
            if start - 1 >= 0 and end <= image.shape[0]:
                line = (image[start - 1, :] + image[end, :]) / 2
                image[start:end, :] = np.tile(line[np.newaxis, :], (end - start, 1))

    return image


def writefile(Path, CT_set, filename):
    # 写入CT图像数据到文件
    full_path = os.path.join(Path, filename)
    with open(full_path, 'wb') as fid:
        fid.write(CT_set.tobytes())


def define(bright_ave, dark_ave, w=1944, h=1536, cut_mean=1744):
    dark_ave[cut_mean:, :] = np.mean(np.mean(dark_ave))
    bright_ave[cut_mean:, :] = np.mean(np.mean(bright_ave))

    # 直方图投影
    bright_ave_nor = bright_ave / np.max(bright_ave)
    dark_ave_nor = dark_ave / np.max(dark_ave)
    size_image = bright_ave_nor.shape

    # 进行垂直灰度投影
    V = np.sum(bright_ave_nor, axis=0) / len(bright_ave_nor)

    # 进行水平灰度投影
    L = np.sum(bright_ave_nor, axis=1) / len(bright_ave_nor)

    return bright_ave_nor, dark_ave_nor, L, V


# 分解图片，获得宽度和高度
def decompose_image(pic, w=1536, h=1944):
    image = np.fromfile(pic, dtype=np.uint16)
    l = len(image)
    frac = np.double(h) / np.double(w)
    w1 = int(np.sqrt(l / frac))
    h1 = int(w1 * frac)
    while l <= w1 * h1:
        h1 -= 1
    return [w1, h1]


def output_image(bright_up, bright_down, dark_up, dark_down, line_1, line_2, directory_in, directory_out,
                 cut_mean=1744):
    directory_in = str(directory_in)
    directory_out = str(directory_out)

    file_list = [directory_in + filename for filename in os.listdir(directory_in) if filename.endswith('.tif')]
    if len(file_list) >= 2:
        filename_bright = random.choice(file_list)

        filename_dark = random.choice([filename for filename in file_list if filename != filename_bright])

        W = []
        H = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 并行计算各个文件的宽度和高度
            futures = {executor.submit(decompose_image, file): file for file in file_list}
            for future in concurrent.futures.as_completed(futures):
                w, h = future.result()
                W.append(w)
                H.append(h)
        # 找出最小的w和h
        w = np.min(W) if W else 0
        h = np.min(H) if H else 0

        bright_ave = np.flipud(np.fromfile(filename_bright, dtype=np.uint16)[:w * h].reshape((w, h)))
        dark_ave = np.flipud(np.fromfile(filename_dark, dtype=np.uint16)[:w * h].reshape((w, h)))

        bright_ave_nor, dark_ave_nor, L, V = define(bright_ave, dark_ave, w, h, cut_mean)

        index_col, index_row, width_col, width_row, index_set_row, index_set_col, L, V = histogramcreate(line_1, line_2,
                                                                                                         V,
                                                                                                         L)

        read_file(index_set_col, index_set_row, dark_ave_nor, dark_up, dark_down, bright_ave_nor, bright_up,
                  bright_down,
                  directory_in, directory_out, w, h)
    else:
        print("Your files are less than 2. Maybe it is an empty directory.")


if __name__ == "__main__":
    start_time = time.time()
    output_image(0.3, 0.05, 0.3, 0.05, 0.15, 0.2, directory_in='../alway/', directory_out='./data/', cut_mean=1744)
    end_time = time.time()
    print("程序运行了 ", end_time - start_time, " 秒")