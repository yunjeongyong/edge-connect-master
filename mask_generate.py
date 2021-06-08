import os
import glob
import numpy as np
import cv2
from random import randint, seed
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image



class MaskGenerator():

    def __init__(self, height=256, width=256, channels=1, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if
                               any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

            # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    # 마스크 생성코드
    def generate_mask(self, i):
        img = np.zeros((self.height, self.width), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)

        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img, (x1, y1), (x2, y2), (1), thickness)

        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img, (x1, y1), radius, (1), -1)

        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1), thickness)

        img = img * 255
        im = Image.fromarray(img.astype(np.uint8))
        im.save('./data/loadroom/mask/mask{}.png'.format(i))
        im.close()



mask = MaskGenerator()
for i in range(27000):
    mask.generate_mask(i)

# a = np.zeros((3, 10, 10), np.uint8)
# a = (np.random.random((10, 10, 3)) * 255.).astype(np.uint8)
# print(a[:, :, 1])