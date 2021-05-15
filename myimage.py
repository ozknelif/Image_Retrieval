import numpy as np
class MyImage:
    def __init__(self, image, filename):
        self.image = image
        self.name = filename
        self.width, self.height = image.size
        self.histogram_r = np.zeros(256, dtype=np.float64)
        self.histogram_g = np.zeros(256, dtype=np.float64)
        self.histogram_b = np.zeros(256, dtype=np.float64)
        self.histogram_h = np.zeros(361, dtype=np.float64)
