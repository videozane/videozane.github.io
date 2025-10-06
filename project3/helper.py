import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.color as skcolor

def get_points(im1, im2=None):
    print('Please select 10 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = plt.ginput(10)
    plt.close()
    if im2 is None:
        return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    plt.imshow(im2)
    p11, p12, p13, p14, p15, p16, p17, p18, p19, p20 = plt.ginput(10)
    plt.close()
    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10], [p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]