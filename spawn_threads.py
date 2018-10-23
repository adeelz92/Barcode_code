import threading

class patchThread(threading.Thread):

    def __init__(self, threadID, name, func, rect, image):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.func = func
        self.rect = rect
        self.image = image

    def run(self):
        self.digits, self.colors = self.func(self.rect, self.image)

    def join(self):
        threading.Thread.join(self)
        return self.digits, self.colors

class rotationThread(threading.Thread):

    def __init__(self, func, image, angle):
        threading.Thread.__init__(self)
        self.func = func
        self.image = image
        self.angle = angle

    def run(self):
        self.digits, self.colors = self.func(self.image, self.angle)

    def join(self):
        threading.Thread.join(self)
        return self.digits, self.colors
