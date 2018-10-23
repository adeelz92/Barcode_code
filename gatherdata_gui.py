# -*- coding: utf-8 -*-
import wx
from DataGather.loadimages import loadImages
from DataGather.get_image_patches import imagePatches
import cv2
import time

IM_INDEX = 0
RECT_INDEX = 0
ANGLE = 0
PATCH_INDEX = 0
REVERT_ANGLE = 0

directory = "./Images"
import os

if not os.path.exists(directory):
    os.makedirs(directory)
    for digit in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for color in ["red", "green", "blue", "black"]:
            os.makedirs(directory + "/" + str(digit) + color)
    os.makedirs(directory + "/" + "nodigit")


class windowClass(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(windowClass, self).__init__(size=(1200, 1000), *args, **kwargs)
        self.path = None
        self.loadimages = loadImages()
        self.image_patches = imagePatches()
        self.images_list = None
        self.basicGUI()

    def basicGUI(self):
        self.panel = wx.Panel(self)

        self.button1 = wx.Button(self.panel, pos=(10, 20), size=(130, 25), label="选择图像文件夹")
        self.path_text = wx.StaticText(self.panel, pos=(150, 25), size=(200, 25), label="未选择文件夹")
        self.a = wx.StaticText(self.panel, pos=(10, 60), size=(40, 20), label="Image")

        self.image_box1 = wx.StaticBitmap(self.panel, pos=(10, 100))
        self.image_box2 = wx.StaticBitmap(self.panel, pos=(300, 100))

        lblList1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "无数字"]
        self.rbox1 = wx.RadioBox(self.panel, label="选择数字", pos=(10, 380), choices=lblList1, majorDimension=1,
                                 style=wx.RA_SPECIFY_ROWS)
        # self.rbox1.Bind(wx.EVT_RADIOBOX, self.onDigitSelection)
        lblList2 = ["红色", "绿色", "蓝色", "黑色"]
        self.rbox2 = wx.RadioBox(self.panel, label="选择颜色", pos=(10, 450), choices=lblList2, majorDimension=1,
                                 style=wx.RA_SPECIFY_ROWS)
        # self.rbox2.Bind(wx.EVT_RADIOBOX, self.onColorSelection)
        self.button2 = wx.Button(self.panel, pos=(635, 520), size=(130, 25), label="下一页")

        self.button1.Bind(wx.EVT_BUTTON, self.open_dialogue)
        self.button2.Bind(wx.EVT_BUTTON, self.detect)

        self.Center()
        self.SetTitle('Data Collection')
        self.Show()

    def open_dialogue(self, e):
        openFileDialog = wx.DirDialog(self.panel, "Choose input directory", "",
                                      wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        openFileDialog.ShowModal()
        self.path = openFileDialog.GetPath()
        self.loadimages.set_folder(folder=self.path)
        self.images_list = self.loadimages.image_list()
        self.path_text.SetLabel(self.path)
        openFileDialog.Destroy()
        if self.images_list is not None:
            self.next_image()

    def detect(self, e):
        global IM_INDEX
        if self.path is not None:
            """
            if IM_INDEX < len(self.images_list):
                digit = self.rbox1.GetStringSelection()
                color = self.rbox2.GetStringSelection()
                if color == "红色":
                    color = "red"
                elif color == "绿色":
                    color = "green"
                elif color == "蓝色":
                    color = "blue"
                elif color == "黑色":
                    color = "black"

                if digit == "无数字":
                    digit = "nodigit"
                    save_path = "Images/" + digit + "/" + str(time.time()) + ".jpg"
                else:
                    save_path = "Images/" + digit + color + "/" + str(time.time()) + ".jpg"
                cv2.imwrite(save_path, self.original_image)
                IM_INDEX += 1
                self.next_image()
            else:
                self.button2.Disable()
                self.a.SetLabel("Finished")
            """
            global PATCH_INDEX
            global RECT_INDEX
            global IM_INDEX
            global ANGLE
            global REVERT_ANGLE

            if self.patch_len is not 0:
                digit = self.rbox1.GetStringSelection()
                color = self.rbox2.GetStringSelection()
                if color == "红色":
                    color = "red"
                elif color == "绿色":
                    color = "green"
                elif color == "蓝色":
                    color = "blue"
                elif color == "黑色":
                    color = "black"

                if digit == "无数字":
                    digit = "nodigit"
                    save_path = "Images/" + digit + "/" + str(time.time()) + ".jpg"
                else:
                    save_path = "Images/" + digit + color + "/" + str(time.time()) + ".jpg"
                #print(save_path)
                cv2.imwrite(save_path, cv2.cvtColor(self.patches[PATCH_INDEX], cv2.COLOR_RGB2BGR))
                PATCH_INDEX += 1
            self.button2.Disable()
            if PATCH_INDEX < self.patch_len:
                self.set_patch(self.patches[PATCH_INDEX])
                self.button2.Enable()
            elif PATCH_INDEX == 16:
                print("Done")
                IM_INDEX += 1
                if IM_INDEX < len(self.images_list):
                    PATCH_INDEX = 0
                    RECT_INDEX = 0
                    ANGLE = 0
                    self.next_image()
                    self.button2.Enable()
                else:
                    self.a.SetLabel("All images finished")
                    ANGLE = 1000
                    RECT_INDEX = 1000
                    self.button2.Disable()
            else:
                if RECT_INDEX < self.rect_len - 1:
                    PATCH_INDEX = 0
                    RECT_INDEX += 1
                    self.patches, self.cropped_image = self.image_patches.get_patches(rect=self.rects[RECT_INDEX],
                                                                                      im_erosion=self.eroded_image)
                    self.patch_len = len(self.patches)
                    if PATCH_INDEX < self.patch_len:
                        self.set_crop(self.cropped_image)
                        self.set_patch(self.patches[PATCH_INDEX])
                    self.button2.Enable()
                else:
                    if ANGLE < 33:
                        PATCH_INDEX = 0
                        RECT_INDEX = 0
                        ANGLE += 3
                        self.rects, self.eroded_image = self.image_patches.rotate_and_detect_rects(
                            image=self.original_image, angle=ANGLE)
                        self.rect_len = len(self.rects)
                        self.patches, self.cropped_image = self.image_patches.get_patches(rect=self.rects[RECT_INDEX],
                                                                                          im_erosion=self.eroded_image)
                        self.patch_len = len(self.patches)
                        if PATCH_INDEX < self.patch_len:
                            self.set_crop(self.cropped_image)
                            self.set_patch(self.patches[PATCH_INDEX])
                        self.button2.Enable()
                    elif REVERT_ANGLE < 33:
                        PATCH_INDEX = 0
                        RECT_INDEX = 0
                        REVERT_ANGLE += 3
                        self.rects, self.eroded_image = self.image_patches.rotate_and_detect_rects(
                            image=self.original_image, angle=360 - REVERT_ANGLE)
                        self.rect_len = len(self.rects)
                        self.patches, self.cropped_image = self.image_patches.get_patches(rect=self.rects[RECT_INDEX],
                                                                                          im_erosion=self.eroded_image)
                        self.patch_len = len(self.patches)
                        if PATCH_INDEX < self.patch_len:
                            self.set_crop(self.cropped_image)
                            self.set_patch(self.patches[PATCH_INDEX])
                        self.button2.Enable()
                    else:
                        IM_INDEX += 1
                        if IM_INDEX < len(self.images_list):
                            PATCH_INDEX = 0
                            RECT_INDEX = 0
                            ANGLE = 0
                            REVERT_ANGLE = 0
                            self.a.SetLabel(self.images_list[IM_INDEX])
                            self.next_image()
                            self.button2.Enable()
                        else:
                            self.a.SetLabel("All images finished")
                            self.button2.Disable()

        else:
            message = wx.MessageBox("Please select image..")

    def set_crop(self, image):

        height, width = image.shape[:2]
        image = wx.Bitmap.FromBuffer(width, height, image)
        self.image_box1.SetBitmap(image)

    def set_patch(self, image):
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = wx.Bitmap.FromBuffer(width, height, image)
        # image = wx.BitmapFromBuffer(width, height, image)
        self.image_box2.SetBitmap(image)

    def next_image(self):
        global IM_INDEX
        """
        self.original_image = cv2.imread(self.images_list[IM_INDEX])
        self.a.SetLabel(os.path.split(self.images_list[IM_INDEX])[1].split(".")[0] + ".jpg")
        self.set_patch(self.original_image)
        """

        global PATCH_INDEX
        self.original_image = self.image_patches.read_image(self.images_list[IM_INDEX])
        self.rects, self.eroded_image = self.image_patches.rotate_and_detect_rects(image=self.original_image,
                                                                                   angle=ANGLE)
        self.rect_len = len(self.rects)
        self.patches, self.cropped_image = self.image_patches.get_patches(rect=self.rects[RECT_INDEX],
                                                                          im_erosion=self.eroded_image)
        self.patch_len = len(self.patches)
        if PATCH_INDEX < self.patch_len:
            self.set_crop(self.cropped_image)
            self.set_patch(self.patches[PATCH_INDEX])

