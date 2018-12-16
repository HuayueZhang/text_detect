# import provider
import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Qt5Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def show_img(path):
    # path = os.path.abspath(path)
    files = os.listdir(path)
    files.sort()

    filtered = []
    for f in files:
        if f.endswith('.png') or f.endswith('.jpg'):
            filtered.append(f)
    for i in range(len(filtered)):
        print(filtered[i])
        pic = mimg.imread(path + filtered[i])
        plt.imshow(pic)
        plt.show()

def mtx_mul():
    A = np.ones([2, 2, 3])
    B = np.ones([3, 6])
    C = np.multiply(A, B)
    print(C)



if __name__ == '__main__':
    # see_data()
    show_img('/home/zhy/pixel_link/test/mtwi_2018/model.ckpt-195671/visual_result/')
    # mtx_mul()