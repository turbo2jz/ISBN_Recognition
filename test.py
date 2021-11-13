import glob
import cv2 as cv
import numpy as np
from isbn_recognition import resize, load_all_kinds_templates, get_rotating_theta, match_templates, get_isbn, \
    get_row_ranges1, get_row_change_num
from utils import foo

train_set_path = "../another/trainset/"
test_set_path = "../another/21exam/"
template_set_path = "./resource/template/"
char_mapping_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X', 'I', 'S', 'B', 'N']

# 获取全部模板
template_set = load_all_kinds_templates(template_set_path, char_mapping_list, resize_w=40, resized_h=50)
# 获取全部测试图片文件路径
img_file_path_list = glob.glob("%s*.*g" % test_set_path, recursive=False)
# 图片计数器
img_cnt = 0
# 识别正确计数器
right_cnt = 0
# 字符总数
chars_sum = 0
# 正确字符总数
right_num_sum = 0

for img_file_path in img_file_path_list:
    # 读入图像
    img = cv.imread(img_file_path)

    # 图像预处理
    # 调整原图大小
    img = resize(img, width=550)
    # 调用cvtColor转换为灰度图
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 中值滤波
    img_gray = cv.medianBlur(img_gray, 3)
    # 转换灰度图为二值图
    ret_whole_pic, img_binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # 计算旋转角度
    thetas = get_rotating_theta(img_binary)
    thetas = 180 * thetas / np.pi - 90
    # 旋转图像
    height, width = img_binary.shape
    model = cv.getRotationMatrix2D((width // 2, height // 2), thetas, 1)
    # 二值图
    img_binary_rotated = cv.warpAffine(img_binary, model, (width, height),
                                       borderMode=cv.INTER_LINEAR, borderValue=cv.BORDER_REPLICATE)
    # 原图
    turn_img = cv.warpAffine(img, model, (width, height),
                             borderMode=cv.INTER_LINEAR, borderValue=cv.BORDER_REPLICATE)
    # 获取行跳变次数
    row_change_nums = get_row_change_num(img_binary_rotated)

    # 计算所在行范围
    tmp_start = 5
    tmp_end = height * 3 // 10
    min_size = 15
    start, end = get_row_ranges1(row_change_nums, tmp_start, tmp_end, 12, 3)
    if (end - start) < min_size:
        start = tmp_start
        end = tmp_end

    # ISBN号所在子图
    sub_img = turn_img[start:end, ]

    # 调整图片大小
    sub_img = resize(sub_img, width=700)
    # 转灰度图
    sub_img_gray = cv.cvtColor(sub_img, cv.COLOR_BGR2GRAY)
    # 转二值图
    sub_img_gray = cv.medianBlur(sub_img_gray, 3)
    ret_part_pic, sub_img_bin = cv.threshold(sub_img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # cv.imwrite('resource/tmp/' + img_file_path.split("\\")[1], sub_img_bin)
    sub_img_height, sub_img_width = sub_img_bin.shape

    # ISBN码预测结果
    res = ""
    # 获得子图全部连通区域
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(sub_img_bin, connectivity=8)
    # 按left,top排序
    stats = sorted(stats, key=lambda s: (s[cv.CC_STAT_LEFT], s[cv.CC_STAT_TOP]))
    # 访问全部连通区域
    for index, stat in enumerate(stats):
        # 获取连通区域特征
        x = stat[cv.CC_STAT_TOP]
        y = stat[cv.CC_STAT_LEFT]
        w = stat[cv.CC_STAT_WIDTH]
        h = stat[cv.CC_STAT_HEIGHT]
        a = stat[cv.CC_STAT_AREA]
        rect_area = w * h
        h_w_rate = h / w
        ca_ra_rate = a / rect_area
        # 排除最大连通区域即图自身
        if h == sub_img_height:
            continue
        # 其他筛选标准：长宽比、连通区域与外接矩形面积比、连通区域面积
        if h_w_rate > 3 or h_w_rate < 1 or ca_ra_rate > 0.9 or a <= 20:
            continue
        sub_char_img = sub_img_bin[x:x + h, y:y + w]
        height, width = sub_char_img.shape
        if height > 0 and width > 0:
            # cv.imwrite("resource/tmp/%d_%d.jpg" % (img_cnt, index), sub_char_img)
            # 模板匹配
            rate, loc = match_templates(resize(sub_char_img, width=40, height=50), template_set)
            # 记录匹配结果
            # res += char_mapping_list[loc] if rate > 0.7 else '_'
            if rate > 0.7:
                res += char_mapping_list[loc]
    # 从图片路径获取ISBN码
    isbn_code = get_isbn(img_file_path)
    # 累计字符总数
    chars_sum += isbn_code.__len__()
    # 输出真值与识别结果对比
    print("No.%3d real: %20s; result: %20s" % (img_cnt + 1, isbn_code, res))
    # 获取数字部分
    res_char_list = list(res)
    num_begin = 0
    for i, c in enumerate(res_char_list):
        if c.isdigit():
            num_begin = i
            break
    isbn_code_num = isbn_code[4:]  # 未处理文件名不包含ISBN情况
    res_num = res[num_begin:]
    # 完全匹配
    if isbn_code_num.__eq__(res_num):
        right_cnt += 1
        right_num_sum += isbn_code_num.__len__()
        print("True")
    # 部分匹配
    else:
        # 真值各类数字按类计数0-9 X
        num_cnt = [0] * 11
        for n in isbn_code_num:
            if n.isdigit():
                num_cnt[int(n)] += 1
            elif n.__eq__("X"):
                num_cnt[10] += 1
        # 查找正确识别的字符并统计数量
        for n in res_num:
            t_n = -1
            if n.isdigit():
                t_n = int(n)
            elif n.__eq__("X"):
                t_n = 10
            if t_n != -1 and num_cnt[t_n] > 0:
                right_num_sum += 1
                num_cnt[t_n] -= 1
        print("False")
    img_cnt += 1

# 正确率及准确率
print("img_num:%4d right_img_num:%4d rate:%.5f" % (img_cnt, right_cnt, right_cnt / img_cnt))
print("num_sum:%4d right_num_sum:%4d rate:%.5f" % (chars_sum - 4 * img_cnt, right_num_sum,
                                                   (right_num_sum / (chars_sum - 4 * img_cnt))))
