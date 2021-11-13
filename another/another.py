# def get_isbn_sub_img1(image, max_size=25, min_size=60):
#     # 设置膨胀腐蚀单元
#     elem1 = cv.getStructuringElement(cv.MORPH_DILATE, (11, 1))
#     elem2 = cv.getStructuringElement(cv.MORPH_ERODE, (1, 2))
#     # 腐蚀、膨胀
#     tmp = cv.erode(img_binary_rotated, elem2, iterations=2)
#     tmp = cv.dilate(tmp, elem1, iterations=3)
#
#     # 获得行像素数组
#     row_pixel_nums = np.zeros((height, 1), dtype=np.uint8)
#     for i in range(height):
#         for j in range(weight):
#             if tmp[i][j] != 0:
#                 row_pixel_nums[i][0] += 1
#
#     # 获取行范围
#     tmp_start = 5
#     tmp_end = height * 3 // 10
#     th = int(np.mean(row_pixel_nums) + np.max(row_pixel_nums)) // 2
#     step_len = 1
#
#     # 迭代计算所在行范围
#     start, end = get_row_ranges(row_pixel_nums, th, tmp_start, tmp_end, min_size)
#     while start == tmp_start or end == tmp_end:
#         if start == tmp_start:
#             tmp_start = start + step_len
#         else:
#             tmp_start = start
#         if end == tmp_end:
#             tmp_end = end - step_len
#         else:
#             tmp_end = end
#         if tmp_end - tmp_start < min_size:
#             break
#         start, end = get_row_ranges(row_pixel_nums, ret_whole_pic, tmp_start, tmp_end, min_size)
#     # 处理大于最大值的情况
#     while end - start > max_size:
#         start, end = get_row_ranges(row_pixel_nums, ret_whole_pic, start + 2, end - 2, min_size)
#
#     if start >= step_len * 7:
#         start -= step_len * 7
#     end -= step_len * 4

# # 检测二值图边缘并膨胀
# # edge = cv.Canny(sub_img_bin, ret_part_pic, 2 * ret_part_pic, apertureSize=3)
# elem4 = cv.getStructuringElement(cv.MORPH_ERODE, (1, 1))
# elem3 = cv.getStructuringElement(cv.MORPH_DILATE, (1, 5))
# tmp1 = cv.erode(sub_img_bin, elem4, iterations=1)
# tmp1 = cv.dilate(tmp1, elem3, iterations=5)
# # plt_image("tmp", tmp, is_gray=True)
#
# # 获取列像素数组
# col_pixel_nums = np.zeros((1, sub_img_weight), np.uint8)
# for j in range(sub_img_weight):
#     for i in range(sub_img_height):
#         if tmp1[i][j] != 0:
#             col_pixel_nums[0][j] += 1
#
# # 获取字符边界
# col_ranges_list = get_col_ranges(col_pixel_nums, threshold=5)
#
# # 截取每个字符
# cnt1 = 0
# res = ""
# for col_ranges in col_ranges_list:
#     char = sub_img[:, col_ranges[0]:col_ranges[1]]
#     height, weight, _ = char.shape
#     if weight == 0:
#         continue
#     char_gray = cv.cvtColor(char, cv.COLOR_BGR2GRAY)
#     _, char_bin = cv.threshold(char_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
#     ret, labels, stats, centroids = cv.connectedComponentsWithStats(char_bin, connectivity=8)
#     for index, stat in enumerate(stats):
#         [x, y, w, h] = stat[0:4]
#         if w == 0 or h == 0:
#             continue
#         if h < height // 3 or h == height:
#             continue
#         sub_char_img = char_bin[x:x + h, y:y + w]
#         height, weight = sub_char_img.shape
#         if height > 0 and weight > 0:
#             cv.imwrite("resource/tmp/%d_%d_%d.jpg" % (cnt, cnt1, index), sub_char_img)
#             rate, loc = match_templates(resize(sub_char_img, width=40, height=50), template_set)
#             # print("%s:%.5f" % (char_mapping_list[loc], rate))
#             res += char_mapping_list[loc]
#     cnt1 += 1
# isbn_code = get_isbn(img_file_name)
# chars_num += isbn_code.__len__()
# print("real: %s; result: %s" % (isbn_code, res))
