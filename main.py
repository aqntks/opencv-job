import cv2
import numpy as np

from hough import *
import easyocr


reader = easyocr.Reader(['en'])
img_path = 'C:/Users/home/Desktop/goodss'


for i in range(0, 12):
    src, con, resize_size, ratio = hough_line_segments(f'{img_path}/test{i}.jpg', show=False)
    # text_img = np.full((int(resize_size * ratio), resize_size, 3), (255, 255, 255), np.uint8)
    text = reader.readtext(src)

    result = ''
    for rect, value, conf in text:
        print(value)
        result += value + '\n'
    # cv2.putText(text_img, result, (0, int(resize_size * ratio)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #
    # con = np.concatenate([con, text_img], axis=1)
    cv2.imshow('concat', con)

    cv2.waitKey()
    cv2.destroyAllWindows()

