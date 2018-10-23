from collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans

class ColorLabler():
    def __init__(self):
        self.colors = OrderedDict({
            0: "Y",
            1: "M",
            2: "C",
            3: "N"
        })
        self.clt = KMeans(2)

    def label(self, image):
        self.clt.fit(image)
        index = np.argmax(self.clt.cluster_centers_[1])
        if self.clt.cluster_centers_[1][index] > 120:
            return self.colors[index]
        else:
            return self.colors[3]