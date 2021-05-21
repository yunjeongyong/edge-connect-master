import glob
import numpy as np
import cv2
from random import randint, seed
from PIL import Image

list = [i for i in glob.glob('./data/images/*')]
print('방 이미지 개수:', len(list))

for idx, room in enumerate(list):
    image = Image.open(room).convert('RGB')
    image = image.resize((256, 256), Image.ANTIALIAS)
    image.save('./data/room/room{}.png'.format(idx+12139))

