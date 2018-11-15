import cv2
import numpy as np
import imutils
from keras import models

classes = {0: "0-K", 1: "0-Y", 2: "0-M", 3: "0-C",
           4: "1-K", 5: "1-Y", 6: "1-M", 7: "1-C",
           8: "2-K", 9: "2-Y", 10: "2-M", 11: "2-C",
           12: "3-K", 13: "3-Y", 14: "3-M", 15: "3-C",
           16: "4-K", 17: "4-Y", 18: "4-M", 19: "4-C",
           20: "5-K", 21: "5-Y", 22: "5-M", 23: "5-C",
           24: "6-K", 25: "6-Y", 26: "6-M", 27: "6-C",
           28: "7-K", 29: "7-Y", 30: "7-M", 31: "7-C",
           32: "8-K", 33: "8-Y", 34: "8-M", 35: "8-C",
           36: "9-K", 37: "9-Y", 38: "9-M", 39: "9-C",
           40: "nodigit"}

digits_codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'nodigit']
colors_codes = ['K', 'Y', 'M', 'C', 'nocolor']

model = models.load_model("../model//first_try.h5")

def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

def show(title, image):
    return 0
    cv2.imshow(title, image)
    cv2.waitKey()


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
            show("Scenario A-1", im_crop)
            x_parts = np.linspace(0, width, 9, dtype=int)
            y_parts = np.linspace(0, height, 3, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
        else:
            show("Scenario A-2", im_crop)
            im_crop = np.delete(im_crop, np.s_[x - 5:x + 5], axis=1)
            height, width = im_crop.shape[:2]
            x_parts = np.linspace(0, width, 9, dtype=int)
            y_parts = np.linspace(0, height, 3, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
    elif height < width / 2 and width / height > 4:
        im_crop = cv2.resize(im_crop, (640, 128), interpolation=cv2.INTER_LANCZOS4)
        im_crop = im_crop[25:-25, 35:-30]
        height, width = im_crop.shape[:2]
        x = int(width / 2)
        check_window = cv2.cvtColor(im_crop[0:height, x - 5:x + 5], cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(check_window, 200, 255, cv2.THRESH_BINARY)[1]
        y = np.average(thresh)
        if y < 250:
            show("Scenario B-1", im_crop)
            x_parts = np.linspace(0, width, 17, dtype=int)
            y_parts = np.linspace(0, height, 2, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
        else:
            show("Scenario B-2", im_crop)
            im_crop = np.delete(im_crop, np.s_[x - 5:x + 5], axis=1)
            height, width = im_crop.shape[:2]
            x_parts = np.linspace(0, width, 17, dtype=int)
            y_parts = np.linspace(0, height, 2, dtype=int)
            digits, colors = detect(x_parts, y_parts, im_crop)
    else:
        im_crop = cv2.resize(im_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        im_crop = im_crop[25:-25, 35:-35]
        show("Scenario C", im_crop)
        height, width = im_crop.shape[:2]
        x_parts = np.linspace(0, width, 5, dtype=int)
        y_parts = np.linspace(0, height, 5, dtype=int)

        digits, colors = detect(x_parts, y_parts, im_crop)

    return digits, colors


def detect(x_parts, y_parts, im_crop):
    digits = []
    colors = []
    kernel = np.ones((2, 2), np.uint8)
    """
    maxIntensity = 255.0
    phi = 1
    theta = 1
    flag = True
    original_image = im_crop
    x = 1
    newImage0 = (maxIntensity / phi) * (im_crop / (maxIntensity / theta)) ** 0.5
    im_crop = np.array(newImage0, dtype=np.uint8)
    while(flag == True):
        hsv = cv2.cvtColor(im_crop, cv2.COLOR_BGR2HSV)
        hsv[...,1] = hsv[...,1]*x
        x = x + 0.25
        im_crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Processed Image', im_crop)
        cv2.waitKey()
        colorfulness = image_colorfulness(im_crop)
        print(colorfulness)
        if colorfulness > 20:
            flag = False
    """
    for i in range(len(y_parts) - 1):
        for j in range(len(x_parts) - 1):
            # original_patch = cv2.imread("D:\Adeel\cnn_digits\Data\\0black\\1539929114.1331685.jpg")
            original_color_patch = im_crop[y_parts[i]:y_parts[i + 1], x_parts[j]:x_parts[j + 1]]
            original_patch = cv2.resize(original_color_patch, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            # patch = cv2.dilate(original_patch, kernel, iterations=2)
            patch_orig = cv2.dilate(original_patch, kernel, iterations=2)
            patch = cv2.cvtColor(patch_orig, cv2.COLOR_BGR2RGB)
            color_patch = np.expand_dims(patch, axis=0)
            # color_patch = np.expand_dims(color_patch, axis=4)
            color_patch = np.divide(color_patch, 255)
            color_patch = color_patch.astype(np.float32)
            digit_pred, color_pred = model.predict(color_patch)
            digit_arg_max = np.argmax(digit_pred[0])
            color_arg_max = np.argmax(color_pred[0])
            digit_prob = digit_pred[0][digit_arg_max] * 100
            color_prob = color_pred[0][color_arg_max] * 100
            digit_code = digits_codes[digit_arg_max]
            color_code = colors_codes[color_arg_max]
            show("Patch", patch)
            # print(digit_pred[0] * 100)
            # print(digit_code)
            if digit_code is not "nodigit" and color_code is not "nocolor" and digit_prob >= 99.99:
                # print(digit_prob)
                # print("Original Detected", digit_code)
                """
                if digit_code == '6' or digit_code == '5' or digit_code == '4':
                    test_patch = cv2.cvtColor(original_color_patch, cv2.COLOR_BGR2GRAY)
                    test_patch = cv2.dilate(test_patch, kernel, iterations=5)
                    test_patch = cv2.resize(test_patch, (32, 32), interpolation=cv2.INTER_LANCZOS4)
                    digit = pytesseract.image_to_string(test_patch, config=(
                    '-l eng --oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'))
                    show("Patch", test_patch)
                    print("Library Detected", digit)
                    if digit == '3':
                        digit_code = digit
                """
                digits.append(digit_code)
                colors.append(color_code)
            else:
                digits = []
                colors = []
                return digits, colors
    return digits, colors


def detect_digits(im, angle):
    # print("Angle", angle)
    im_rotate = imutils.rotate(im, angle=angle)
    im_resize = cv2.resize(im_rotate, (1600, 1600), interpolation=cv2.INTER_LANCZOS4)
    kernel = np.ones((2, 2), np.uint8)
    im_erosion = cv2.erode(im_resize, kernel, iterations=1)
    gray = cv2.cvtColor(im_erosion, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 4)
    # show("BW", thresh)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if not cv2.isContourConvex(cnt)]
    rects = [detect_rects(cnt) for cnt in contours]
    rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
    rects = rects[:10]

    for i, rect in enumerate(rects):
        digits, colors = crop(rect, im_erosion)
        if len(digits) == 16:
            return digits, colors
    return [], []


if __name__ == '__main__':

    im = cv2.imread("../test/test13.jpg")

    for angle in range(0, 40, 2):
        digits_from_1, colors_from_1 = detect_digits(im, angle)

        if len(digits_from_1) is 16:
            print(digits_from_1)
            print(colors_from_1)
            break

        digits_from_2, colors_from_2 = detect_digits(im, 360 - angle)

        if len(digits_from_2) is 16:
            print(digits_from_2)
            print(colors_from_2)
            break
