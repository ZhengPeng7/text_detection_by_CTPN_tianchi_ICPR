import cv2
from math import fabs, sin, cos, radians
import numpy as np
from scipy.stats import mode


def get_img_rot_broa(img, degree=45, filled_color=-1):
    """
    Desciption:
            Get img rotated a certain degree anticlockwisely,
        and use some color to fill 4 corners of the new img.
    """

    # Get the filled_color for the addtional four corners
    if filled_color == -1:
        filled_color = mode([img[0, 0], img[0, -1],
                             img[-1, 0], img[-1, -1]]).mode[0]
    if len(img.shape) == 2:
        filled_color = tuple([int(i) for i in range(3)])
    else:
        filled_color = tuple([int(i) for i in filled_color])

    height, width = img.shape[:2]

    # the shape after rotation
    height_new = int(width * fabs(sin(radians(degree))) +
                     height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) +
                    width * fabs(cos(radians(degree))))

    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2

    # Pay attention to the type of elements of filler_color, which should be
    # the int in pure python, instead of those in numpy.
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=filled_color)
    # fill four corners
    mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
    mask[:] = 0
    seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
                   (width_new - 1, height_new - 1)]
    for i in seed_points:
        cv2.floodFill(img_rotated, mask, i, filled_color)

    return img_rotated
