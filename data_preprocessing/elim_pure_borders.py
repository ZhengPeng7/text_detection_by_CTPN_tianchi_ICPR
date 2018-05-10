import cv2


def elim_pure_borders(img):
    """
    有待改进：如果图片是斜着的，需要先找最小外接矩形，扶正，再找外接矩形，提取。
    Description: To cut out the img-region from the whole img.
    """
    # eliminate the white borders
    img_ori = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(255 - img, 200, 255, cv2.THRESH_BINARY)[1]

    for i in range(img.shape[0]):
        if img[i, :].any():
            break
    top = i
    for i in range(img.shape[0]-1, -1, -1):
        if img[i, :].any():
            break
    bottom = i
    for i in range(img.shape[1]):
        if img[:, i].any():
            break
    left = i
    for i in range(img.shape[1]-1, -1, -1):
        if img[:, i].any():
            break
    right = i
    img = img[top:bottom, left:right]
    img_ori = img_ori[top:bottom, left:right]

    # eliminate the black borders
    img = cv2.threshold(255 - img, 200, 255, cv2.THRESH_BINARY)[1]
    for i in range(img.shape[0]):
        if img[i, :].any():
            break
    top = i
    for i in range(img.shape[0]-1, -1, -1):
        if img[i, :].any():
            break
    bottom = i
    for i in range(img.shape[1]):
        if img[:, i].any():
            break
    left = i
    for i in range(img.shape[1]-1, -1, -1):
        if img[:, i].any():
            break
    right = i
    img_ori = img_ori[top:bottom, left:right]
    return img_ori
