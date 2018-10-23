import cv2
import numpy as np
import pytesseract
import imutils

from spawn_threads import patchThread, rotationThread
from color_labler import ColorLabler


def detect_rects(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # print(approx)
    if len(approx) == 4:
        return cv2.boundingRect(approx)
    else:
        return (0, 0, 0, 0)


def crop(rect, im_erosion):
    im_crop = im_erosion[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    height, width = im_crop.shape[:2]
    if height < width / 2 and width / height < 4:
        im_crop = cv2.resize(im_crop, (512, 128), interpolation=cv2.INTER_LANCZOS4)
        im_crop = im_crop[25:-25, 65:-65]
        height, width = im_crop.shape[:2]
        x = int(width / 2)
        check_window = cv2.cvtColor(im_crop[0:height, x - 5:x + 5], cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(check_window, 200, 255, cv2.THRESH_BINARY)[1]
        y = np.average(thresh)
        if y < 250:
            x_parts = np.linspace(0, width, 9, dtype=int)
            y_parts = np.linspace(0, height, 3, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
        else:
            im_crop = np.delete(im_crop, np.s_[x - 5:x + 5], axis=1)
            height, width = im_crop.shape[:2]
            x_parts = np.linspace(0, width, 9, dtype=int)
            y_parts = np.linspace(0, height, 3, dtype=int)
            digits,colors = detect(x_parts, y_parts, im_crop)
    elif height < width / 2 and width / height > 4:
        im_crop = cv2.resize(im_crop, (640, 128), interpolation=cv2.INTER_LANCZOS4)
        im_crop = im_crop[25:-25, 35:-35]
        height, width = im_crop.shape[:2]
        x = int(width / 2)
        check_window = cv2.cvtColor(im_crop[0:height, x - 5:x + 5], cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(check_window, 200, 255, cv2.THRESH_BINARY)[1]
        y = np.average(thresh)
        if y < 250:
            x_parts = np.linspace(0, width, 17, dtype=int)
            y_parts = np.linspace(0, height, 2, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
        else:
            im_crop = np.delete(im_crop, np.s_[x - 5:x + 5], axis=1)
            height, width = im_crop.shape[:2]
            x_parts = np.linspace(0, width, 17, dtype=int)
            y_parts = np.linspace(0, height, 2, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
    else:
        im_crop = cv2.resize(im_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        im_crop = im_crop[25:-25, 35:-35]
        height, width = im_crop.shape[:2]
        x_parts = np.linspace(0, width, 5, dtype=int)
        y_parts = np.linspace(0, height, 5, dtype=int)

        digits, colors = detect(x_parts, y_parts, im_crop)

    return digits, colors


def detect(x_parts, y_parts, im_crop):
    digits = []
    colors = []
    kernel = np.ones((2, 2), np.uint8)
    for i in range(len(y_parts) - 1):
        for j in range(len(x_parts) - 1):
            original_patch = im_crop[y_parts[i]:y_parts[i + 1], x_parts[j]:x_parts[j + 1]]
            color_patch = cv2.resize(original_patch, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            patch = cv2.cvtColor(original_patch, cv2.COLOR_BGR2GRAY)
            patch = cv2.dilate(patch, kernel, iterations=4)
            patch = cv2.resize(patch, (28, 28), interpolation=cv2.INTER_LANCZOS4)

            #color_patch = cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB)
            #cv2.imshow("Color", color_patch)
            #cv2.waitKey()
            color_patch = color_patch.reshape((color_patch.shape[0] * color_patch.shape[1], 3))


            digit = pytesseract.image_to_string(patch, config=(
                '-l eng --oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'))
            if (digit.isdigit()):
                CL = ColorLabler()
                color = CL.label(color_patch)
                digits.append(digit)
                colors.append(color)
            else:
                digits = []
                colors = []
                return digits, colors
    return digits, colors

def detect_digits(im, angle):
    #print("Angle", angle)
    im_rotate = imutils.rotate(im, angle=angle)
    im_resize = cv2.resize(im_rotate, (1600, 1600), interpolation=cv2.INTER_LANCZOS4)
    kernel = np.ones((2, 2), np.uint8)
    im_erosion = cv2.erode(im_resize, kernel, iterations=1)
    gray = cv2.cvtColor(im_erosion, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if not cv2.isContourConvex(cnt)]
    rects = [detect_rects(cnt) for cnt in contours]
    rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
    rects = rects[:10]
    thread = []

    for i, rect in enumerate(rects):
        thread.append(patchThread(threadID=i + 1, name="Patch-{}".format(i), func=crop, rect=rect, image=im_erosion))
        thread[i].start()
    for i in range(len(thread)):
        digits, colors = thread[i].join()
        if len(digits) == 16:
            return digits, colors
    return [],[]

def run(path):
    im = cv2.imread(path)
    for angle in range(0, 360, 3):
        thread1 = rotationThread(func=detect_digits, image=im, angle=angle)
        thread1.setDaemon(True)
        thread2 = rotationThread(func=detect_digits, image=im, angle=360 - angle)
        thread2.setDaemon(True)

        thread1.start()
        thread2.start()

        digits_from_1, colors_from_1 = thread1.join()
        digits_from_2, colors_from_2 = thread2.join()

        if len(digits_from_1) is 16:
            return digits_from_1, colors_from_1
        elif len(digits_from_2) is 16:
            return digits_from_2, colors_from_2