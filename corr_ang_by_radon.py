import cv2
from skimage import transform, morphology
import numpy as np
import get_img_rot_broa


def corr_ang_by_radon(image):
    """
    Description: use the method of radon transformation to straighten the img.
    :param image: src
    :return: img_rotated and rotating_angle
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.Canny(img_gray, 60, 120)
    for i in bw:
        for j in range(len(i)):
            if 255 == i[j]:
                i[j] = 1
    bw = morphology.skeletonize(bw)   # optional
    theta = list(range(-90, 90))
    R = transform.radon_transform.radon(bw, theta)
    R1 = np.max(R, axis=0)
    theta_max = 90
    while theta_max > 50 or theta_max < -50:
        theta_max = np.argwhere(R1 == np.max(R1))[0]
        R1[theta_max] = 0
        theta_max -= 91
    img_rotated = get_img_rot_broa.get_img_rot_broa(img, -theta_max)
    rotating_angle = theta_max

    return img_rotated, rotating_angle
