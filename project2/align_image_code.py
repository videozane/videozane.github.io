import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.color as skcolor


def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1./dscale)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi, resize=True)
    return im1, dtheta
    
def match_img_size(im1, im2):
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    min_h = min(h1, h2)
    min_w = min(w1, w2)

    start_h1 = (h1 - min_h) // 2
    start_w1 = (w1 - min_w) // 2
    
    start_h2 = (h2 - min_h) // 2
    start_w2 = (w2 - min_w) // 2

    im1_cropped = im1[start_h1 : start_h1 + min_h, start_w1 : start_w1 + min_w, :]
    im2_cropped = im2[start_h2 : start_h2 + min_h, start_w2 : start_w2 + min_w, :]
    
    assert im1_cropped.shape == im2_cropped.shape, f"Image shapes do not match after cropping: {im1_cropped.shape} vs {im2_cropped.shape}"
    return im1_cropped, im2_cropped

def standardize_to_rgb(image):
    if image.ndim == 2: 
        return skcolor.gray2rgb(image)
    elif image.shape[2] == 4:
        return image[:, :, :3] 
    elif image.shape[2] == 1:
        return skcolor.gray2rgb(image.squeeze(axis=2))
    elif image.shape[2] == 2: 
        return skcolor.gray2rgb(image[:, :, 0])
    
    return image

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1 = standardize_to_rgb(im1)
    im2 = standardize_to_rgb(im2)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!
    pass
