import glob
import cv2 as cv
import numpy as np


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    :param image: 图像
    :param width: 宽度
    :param height: 高度
    :param inter:
    :return: resized 缩放后图像
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        size = (int(w * r), height)
    else:
        r = width / float(w)
        size = (width, int(h * r))
    resized = cv.resize(image, size, interpolation=inter)
    return resized


def get_isbn(file_path):
    """
    :param file_path: 待识别图像路径
    :return: isbn
    """
    isbn = ""
    # file_path需为glob.glob()利用匹配方式读入的
    file_name = file_path.split("\\")[1]
    char_list = list(file_name)
    for c in char_list:
        if c.isdigit() or c in ['I', 'S', 'B', 'N', 'X']:
            isbn += c
    return isbn


def get_row_change_num(image_bin):
    """
    :param image_bin: 二值图
    :return: 行跳变次数数组
    """
    height, width = image_bin.shape
    row_change_nums = np.zeros((height, 1), dtype=np.uint8)
    for i in range(height):
        before = image_bin[i][0]
        for j in range(1, width):
            if image_bin[i][j] != before:
                row_change_nums[i][0] += 1
                before = image_bin[i][j]
    return row_change_nums


def get_rotating_theta(image, th=180):
    """
    :param image: 图像
    :param th: 直线长度阈值
    :return: thetas
    """
    # 对输入图像在y方向上求一阶导数
    edge = cv.Sobel(image, ddepth=cv.CV_8U, dx=0, dy=1, ksize=3)
    # 强化
    edge = cv.convertScaleAbs(edge)
    # 获取图像中的直线
    line_list = cv.HoughLines(edge, rho=1, theta=np.pi / 180, threshold=th)
    thetas = 0
    # 直线列表为空的情况
    if line_list is None:
        thetas = np.pi / 2
    else:
        # 累加直线角度
        for line in line_list:
            thetas += line[0][1]
        # 处理没有直线的情况（不确定会否到达该分支）
        if len(line_list) == 0:
            thetas = np.pi / 2
        # 除以直线数量获得最终结果
        else:
            thetas /= len(line_list)
    return thetas


def get_row_ranges(row_pixel_nums, threshold, min_row, max_row, min_size, min_pixels=0):
    """
    :param row_pixel_nums: 行像素数组
    :param threshold: 二值化阈值
    :param min_row: 最小行
    :param max_row: 最大行
    :param min_size: 最小差值
    :param min_pixels: 边界最小判断像素值
    :return: start,end
    """
    start = min_row
    end = max_row
    # 均值模糊
    dst = cv.blur(row_pixel_nums, (3, 3))
    # Canny边缘检测
    dst = cv.Canny(dst, threshold, threshold * 2, apertureSize=3)

    # 查找起始行
    for i in range(min_row, max_row, 1):
        if dst[i][0] != min_pixels:
            start = i
            break
    # 查找结束行
    for i in range(max_row, min_row - 1, -1):
        if dst[i][0] != min_pixels:
            end = i
            break
    # 处理范围小于预设值的情况，修改阈值，再次查找
    if abs(end - start) < min_size:
        threshold -= 5
        if threshold < 0:
            return min_row, max_row
        else:
            return get_row_ranges(row_pixel_nums, threshold, min_row, max_row, min_size)
    return start, end


def get_row_ranges1(row_change_nums, min_row, max_row, th1, th2):
    """
    :param row_change_nums: 行像素数组
    :param min_row: 最小行
    :param max_row: 最大行
    :param th1: 字符上下边界跳变阈值
    :param th2: 条码上边界跳变次数阈值
    :return: start,end
    """
    gap = 3
    start, end = min_row, max_row
    # 查找起始行
    for i in range(min_row, max_row, 1):
        if row_change_nums[i][0] > th1:
            t_start = i - gap
            if t_start > 0:
                start = t_start
            else:
                start = i
            break
    # 查找结束行
    t_end = -1
    for i in range(max_row, start - 1, -1):
        if row_change_nums[i][0] < th2 and t_end == -1:
            t_end = i
        if row_change_nums[i][0] > th1 and t_end != -1:
            end = t_end - gap

    return start, end


def get_col_ranges(col_pixel_nums, threshold=0):
    """
    :param col_pixel_nums: 列像素数组
    :param threshold: 阈值
    :return: char_col_range_list
    """
    char_col_range_list = []
    _, width = col_pixel_nums.shape
    for j in range(1, width - 1, 1):
        if col_pixel_nums[0][j] > threshold >= col_pixel_nums[0][j - 1]:
            char_col_range_list.append(j - 1)
        elif col_pixel_nums[0][j] > threshold >= col_pixel_nums[0][j + 1]:
            char_col_range_list.append(j + 1)
    # 处理仅获得奇数个边界的情况
    if len(char_col_range_list) % 2 != 0:
        char_col_range_list.append(width - 1)
    return np.array(char_col_range_list).reshape((-1, 2))


def load_all_kinds_templates(tmpl_set_path, c_mapping_list, resize_w, resized_h):
    """
    :param tmpl_set_path: 模板路径
    :param c_mapping_list: 字符映射列表
    :param resize_w: 模板宽
    :param resized_h: 模板高
    :return:
    """
    tmpls = []
    for c in c_mapping_list:
        same_tmpls = []
        path = "%s%s/" % (tmpl_set_path, c)
        tmpl_name_list = glob.glob("%s*.*g" % path, recursive=False)
        for tmpl_name in tmpl_name_list:
            same_tmpls.append(resize(cv.imread(tmpl_name, 0), width=resize_w, height=resized_h))
        tmpls.append(same_tmpls)
    return tmpls


def match_same_kind_templates(char_image, same_kind_tmpl_list):
    """
    :param char_image: 待匹配字符
    :param same_kind_tmpl_list: 同类模板列表
    :return:
    """
    # 与该类别中全部模板进行匹配获得最大值
    match_degree = -1
    for template in same_kind_tmpl_list:
        # cv.TM_CCOEFF_NORMED 归一化相关系数匹配法
        rate_list = cv.matchTemplate(char_image, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(rate_list)
        if max_val >= match_degree:
            match_degree = max_val
    return match_degree


def match_templates(char_image, tmpl_list):
    """
    :param char_image: 待匹配字符
    :param tmpl_list: 各类模板列表的列表
    :return:
    """
    # 与全部模板进行匹配获得最大值
    match_degree = -1
    res_loc = 0
    for idx, tmpl in enumerate(tmpl_list):
        tmp_match_degree = match_same_kind_templates(char_image, tmpl)
        if tmp_match_degree >= match_degree:
            match_degree = tmp_match_degree
            res_loc = idx
    return match_degree, res_loc
