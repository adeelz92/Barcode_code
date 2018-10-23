import glob

class loadImages():
    def __init__(self):
        self.folder = None

    def set_folder(self, folder):
        self.folder = folder

    def image_list(self):
        images_list = glob.glob(self.folder+"/*")
        print(images_list)
        return images_list