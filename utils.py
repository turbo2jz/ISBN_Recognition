import numpy as np
from matplotlib import pyplot as plt


def plt_hist(image):
    """
    :param image: 灰度图像
    """
    plt.title("image_hist")
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def plt_image(image_name, image, is_gray=False):
    """
    :param image_name: 图像名
    :param image: 图像
    :param is_gray: 是否为灰度图
    """
    plt.figure(image_name)
    plt.title(image_name)
    if is_gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()


def plt_row_pixel_nums(title, row_pixel_nums):
    """
    :param title: 标题
    :param row_pixel_nums: 行非零像素总和数组
    """
    plt.figure(title)
    plt.title(title)
    tmp = np.squeeze(row_pixel_nums).tolist()
    plt.barh(range(1, len(tmp) + 1), tmp)
    # plt.show()
    plt.savefig('resource/tmp/' + title + '.png')
