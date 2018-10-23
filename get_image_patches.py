import cv2
import imutils
import numpy as np
import pytesseract


class imagePatches():
    def __init__(self):
        pass

    def read_image(self, path):
        image = cv2.imread(path)
        return image

    def rotate_and_detect_rects(self, image, angle):
        im_rotate = imutils.rotate(image, angle=angle)
        im_resize = cv2.resize(im_rotate, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        kernel = np.ones((2, 2), np.uint8)
        im_erosion = cv2.erode(im_resize, kernel, iterations=1)
        gray = cv2.cvtColor(im_erosion, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 10)
        # show("Edges", thresh)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if not cv2.isContourConvex(cnt)]

        def detect_rects(c):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # print(approx)
            if len(approx) == 4:
                return cv2.boundingRect(approx)
            else:
                return (0, 0, 0, 0)

        rects = [detect_rects(cnt) for cnt in contours]
        rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
        rects = rects[:10]
        return rects, im_erosion

    def get_patches(self, rect, im_erosion):
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
                patches = self.detect(x_parts, y_parts, im_crop)
            else:
                im_crop = np.delete(im_crop, np.s_[x - 5:x + 5], axis=1)
                height, width = im_crop.shape[:2]
                x_parts = np.linspace(0, width, 9, dtype=int)
                y_parts = np.linspace(0, height, 3, dtype=int)
                patches = self.detect(x_parts, y_parts, im_crop)
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
                patches = self.detect(x_parts, y_parts, im_crop)
            else:
                im_crop = np.delete(im_crop, np.s_[x - 5:x + 5], axis=1)
                height, width = im_crop.shape[:2]
                x_parts = np.linspace(0, width, 17, dtype=int)
                y_parts = np.linspace(0, height, 2, dtype=int)
                patches = self.detect(x_parts, y_parts, im_crop)
        else:
            im_crop = cv2.resize(im_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            im_crop = im_crop[25:-25, 35:-35]
            height, width = im_crop.shape[:2]
            x_parts = np.linspace(0, width, 5, dtype=int)
            y_parts = np.linspace(0, height, 5, dtype=int)

            patches = self.detect(x_parts, y_parts, im_crop)

        return patches, cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)

    def detect(self, x_parts, y_parts, im_crop):
        patches = []
        kernel = np.ones((2, 2), np.uint8)
        for i in range(len(y_parts) - 1):
            for j in range(len(x_parts) - 1):
                original_patch = im_crop[y_parts[i]:y_parts[i + 1], x_parts[j]:x_parts[j + 1]]
                color_patch = cv2.resize(original_patch, (28, 28), interpolation=cv2.INTER_LANCZOS4)
                patch = cv2.cvtColor(original_patch, cv2.COLOR_BGR2GRAY)
                patch = cv2.dilate(patch, kernel, iterations=4)
                patch = cv2.resize(patch, (28, 28), interpolation=cv2.INTER_LANCZOS4)

                digit = pytesseract.image_to_string(patch, config=(
                    '-l eng --oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'))
                if (digit.isdigit()):
                    patches.append(cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB))
                else:
                    return patches
        return patches
