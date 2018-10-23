import cv2
import numpy as np
import imutils
import time
from spawn_threads import rotationThread
from keras import models
import tensorflow as tf

classes = {0: "0-K", 1: "0-Y", 2: "0-M", 3: "0-C",
           4: "1-K", 5: "1-Y", 6: "1-M", 7: "1-C",
           8: "2-K", 9: "2-Y", 10: "2-M", 11: "2-C",
           12: "3-K", 13: "4-Y", 14: "5-M", 15: "6-C",
           16: "4-K", 17: "4-Y", 18: "4-M", 19: "4-C",
           20: "5-K", 21: "5-Y", 22: "5-M", 23: "5-C",
           24: "6-K", 25: "6-Y", 26: "6-M", 27: "6-C",
           28: "7-K", 29: "7-Y", 30: "7-M", 31: "7-C",
           32: "8-K", 33: "8-Y", 34: "8-M", 35: "8-C",
           36: "9-K", 37: "9-Y", 38: "9-M", 39: "9-C",
           40: "nodigit"}

model = models.load_model("model//first_try.h5")
graph = tf.get_default_graph()

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
        im_crop = im_crop[25:-25, 35:-30]
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
    for i in range(len(y_parts) - 1):
        for j in range(len(x_parts) - 1):
            # original_patch = cv2.imread("D:\Adeel\cnn_digits\Data\\0black\\1539929114.1331685.jpg")
            original_patch = im_crop[y_parts[i]:y_parts[i + 1], x_parts[j]:x_parts[j + 1]]
            original_patch = cv2.resize(original_patch, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            color_patch = cv2.cvtColor(original_patch, cv2.COLOR_BGR2RGB)
            color_patch = np.expand_dims(color_patch, axis=0)
            color_patch = np.divide(color_patch, 255)
            with graph.as_default():
                pred = model.predict(color_patch)
            arg_max = np.argmax(pred)
            prob = pred[0][arg_max] * 100
            code = classes[arg_max]
            #show("Patch", original_patch)
            if code is not "nodigit":
                # print(prob)
                digit, color = code.split("-")
                # print(digit)
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
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 4)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if not cv2.isContourConvex(cnt)]
    rects = [detect_rects(cnt) for cnt in contours]
    rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
    rects = rects[:10]

    for i, rect in enumerate(rects):
        digits, colors = crop(rect, im_erosion)
        if len(digits) == 16:
            return digits, colors
    return [],[]

def run(path):
    im = cv2.imread(path)
    t = time.time()
    for angle in range(0, 12, 2):
        thread1 = rotationThread(func=detect_digits, image=im, angle=angle)
        thread1.setDaemon(True)
        thread2 = rotationThread(func=detect_digits, image=im, angle=360 - angle)
        thread2.setDaemon(True)

        thread1.start()
        thread2.start()

        digits_from_1, colors_from_1 = thread1.join()
        digits_from_2, colors_from_2 = thread2.join()

        if len(digits_from_1) is 16:
            ti = time.time() - t
            print(ti)
            return digits_from_1, colors_from_1, ti
        elif len(digits_from_2) is 16:
            ti = time.time() - t
            return digits_from_2, colors_from_2, ti