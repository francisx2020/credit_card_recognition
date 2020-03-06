"""
    作者：fly2020
    时间：2020/02
    功能：实现指定银行卡号的识别
"""
import cv2
import numpy as np
from copy import copy

# 图片路径，模板路径
img_path = "./credit_card_04.png"
temp_path = "./template.png"


# 模板数字排序函数
def sort_contours(cnts, method="left-right"):
    reverse = False
    i = 0
    if method == "right-left" or method == "bottom-top":
        reverse = True
    if method == "top-bottom" or method == "bottom-top":
        i = 1
    # 用矩形将模板图片中每个数字的信息存储在bounding_box中
    bounding_box = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_box = zip(*sorted(zip(cnts, bounding_box),
                                     key=lambda x: x[1][i], reverse=reverse))
    return cnts, bounding_box


# 读取图片
def cv_imshow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 导入图片的尺寸进行调整
def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    elif width is None:
        rate = height / float(h)
        size = (int(w * rate), height)
    else:
        rate = width / float(w)
        size = (width, int(h * rate))

    resized = cv2.resize(img, size, interpolation=inter)
    return resized


# 模板图片处理
def temp_process(file_path):
    img = cv2.imread(file_path)
    # cv_imshow(name="temp", img=img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # cv_imshow('thresh', res)

    """
    cv2.findContours()函数接受的参数为二值图，即黑白的(不是灰度图)
    cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
    返回的list中每个元素都是图像中的一个轮廓
    """
    _, cnts, hierarchy = cv2.findContours(copy(res), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
    # cv_imshow('draw', img)
    cnts, _ = sort_contours(cnts, method="left-right")

    temp_digits = {}
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        interest = res[y:y+h, x:x+w]
        interest = cv2.resize(interest, (57, 88))
        temp_digits[i] = interest

    return temp_digits


# 卡号识别
def card_recog(img_path, tem_path):
    image = cv2.imread(img_path)
    # cv_imshow(name='orig', img=image)
    image = resize(image, width=300)
    # cv_imshow('reszie', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv_imshow('gray', gray)

    # cv2.MORPH_TOPHAT, 突出边界区域
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    to_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel=kernel1)
    # cv_imshow('to_hat', to_hat)

    # 突出明亮部分轮廓
    grand_x = cv2.Sobel(to_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grand_x = np.absolute(grand_x)
    (min, max) = (np.min(grand_x), np.max(grand_x))
    grand_x = (255 * ((grand_x - min) / (max - min)))
    grand_x = grand_x.astype("uint8")
    # cv_imshow('grand_x', grand_x)

    # 先进行闭操作，把相邻位置突出
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    morph1 = cv2.morphologyEx(grand_x, cv2.MORPH_CLOSE, kernel=kernel2)
    # cv_imshow('morph1', morph1)
    # 再进行二值化
    _, thresh = cv2.threshold(morph1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv_imshow('thresh', thresh)
    # 再进行闭操作
    morph2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel2)
    # cv_imshow('morph2', morph2)

    # 计算轮廓
    _, cnts, hierarchy = cv2.findContours(morph2.copy(),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    location = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        rate = w / float(h)
        # 根据实际识别的卡片尺寸，选择合适的区域
        if 2 < rate < 4.5:
            if (30 < w < 60) and (5 < h < 25):
                location.append((x, y, w, h))
    location = sorted(location, key=lambda b: b[0])

    # 遍历每一个提取后的轮廓数字
    output = []
    for (i, (xx, yy, ww, hh)) in enumerate(location):
        out_group = []
        # 提取每一个轮廓中的单个数字
        group = gray[yy-5:yy+hh+5, xx-5:xx+ww+5]
        _, group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 每一小组的轮廓
        _, cnts, hierarchy = cv2.findContours(group.copy(),
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits_cnts, bounding_box = sort_contours(cnts, method="left-right")

        # 将每个一数字修改尺寸后与模板匹配
        for c in digits_cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            interest = group[y:y+h, x:x+w]
            interest = cv2.resize(interest, (57, 88))
            # 计算匹配得分p并保存在列表中
            scores = []
            digits = temp_process(file_path=tem_path)
            for key, value in digits.items():
                # 模板匹配
                result = cv2.matchTemplate(interest, value, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            # 得到该组数字
            out_group.append(str(np.argmax(scores)))

        # 将得到的匹配数值画在原图上
        cv2.rectangle(image, (xx-5, yy-5), (xx+ww+5, yy+hh+5), (0, 0, 255), 1)
        cv2.putText(image, "".join(out_group), (xx, yy-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)

        output.append(out_group)

    cv_imshow(name="temp", img=image)


if __name__ == "__main__":
    card_recog(img_path=img_path, tem_path=temp_path)