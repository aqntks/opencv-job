import numpy as np
import cv2
import math
from sympy import Derivative, symbols, solve


def hough_lines(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLines(edge, 1, math.pi / 180, 250)

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            alpha = 1000
            pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
            pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def hough_line_segments(img_path, show=False):
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ori = src.copy()
    ratio = src.shape[0] / src.shape[1]

    # src = cv2.resize(src, (int(size*ratio), size))

    # for i in range(src.shape[0]):
    #     for y in range(src.shape[1]):
    #         if src[i][y] > 180:
    #             src[i][y] = 255


    h = src.shape[0]
    w = src.shape[1]

    if src is None:
        print('Image load failed!')
        return

    edge = cv2.Canny(src, 50, 150)
    # edge = cv2.Canny(src, 20, 40)
    lines = cv2.HoughLinesP(edge, 1, math.pi / 180, 4, minLineLength=42, maxLineGap=3)

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # for i in range(lines.shape[0]):
    #     pt1 = (lines[i][0][0], lines[i][0][1])
    #     pt2 = (lines[i][0][2], lines[i][0][3])
    #
    #     cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    #     cv2.line(src, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    if lines is not None:
        for i in range(lines.shape[0]):
            x1, y1, x2, y2 = lines[i][0][0], h - lines[i][0][1], lines[i][0][2], h - lines[i][0][3]
            if x2 - x1 != 0:
                m = (y2 - y1) / (x2 - x1)
                x, b = symbols('x'), symbols('b')
                fx = m * x + b
                b = y1 - m * x1

            if lines[i][0][1] == lines[i][0][3]:
                pt1 = (0, lines[i][0][1])
                pt2 = (w, lines[i][0][3])
            elif abs(lines[i][0][2] - lines[i][0][0]) > abs(lines[i][0][3] - lines[i][0][1]):
                x = 0
                startY = m * x + b
                x = w
                finishY = m * x + b

                startY = int(h - startY)
                finishY = int(h - finishY)

                pt1 = (0, startY)
                pt2 = (w, finishY)
            # elif lines[i][0][0] == lines[i][0][2]:
            #     pt1 = (lines[i][0][0], 0)
            #     pt2 = (lines[i][0][2], h)
            # elif abs(lines[i][0][2] - lines[i][0][0]) < abs(lines[i][0][3] - lines[i][0][1]):
            #     y = h
            #     startX = (y - b) / m
            #     y = 0
            #     finishX = (y - b) / m
            #
            #     startX = int(startX)
            #     finishX = int(finishX)
            #
            #     pt1 = (startX, 0)
            #     pt2 = (finishX, h)
            else:
                pt1 = (0, 0)
                pt2 = (0, 0)

            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(src, pt1, pt2, (255, 255, 255), 5, cv2.LINE_AA)

    resize_size = 250 if src.shape[1] > 250 else src.shape[1]

    ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    con = np.concatenate([ori, np.concatenate([src, dst], axis=1)], axis=1)
    con = cv2.resize(con, dsize=(resize_size * 3, int(resize_size * ratio)))
    if show:

        cv2.imshow('concat', cv2.resize(con, dsize=(resize_size * 3, int(resize_size * ratio))))

        cv2.waitKey()
        cv2.destroyAllWindows()

    return src, con, resize_size, ratio


if __name__ == '__main__':

    img_path = 'C:/Users/home/Desktop/goodss'
    # hough_lines(img_path)

    for i in range(0, 12):
        hough_line_segments(f'{img_path}/test{i}.jpg', show=True)


