import numpy as np
import cv2

def escolher_offset(source, target, mask):
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cut = source[y_min:y_max+1, x_min:x_max+1]
    mask_cut = mask[y_min:y_max+1, x_min:x_max+1]

    cut = cv2.cvtColor(cut, cv2.COLOR_RGB2BGR)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

    h, w = cut.shape[:2]
    click = [-1, -1]

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click[0] = x
            click[1] = y

    temp = target.copy()
    cv2.namedWindow("posicione")
    cv2.setMouseCallback("posicione", callback)

    while True:
        if click[0] != -1:
            temp = target.copy()
            y0 = click[1]
            x0 = click[0]
            y1 = y0 + h
            x1 = x0 + w
            if y0 >= 0 and x0 >= 0 and y1 <= target.shape[0] and x1 <= target.shape[1]:
                temp[y0:y1, x0:x1] = np.where(mask_cut[...,None], cut, temp[y0:y1, x0:x1])
        cv2.imshow("posicione", temp)
        k = cv2.waitKey(1)
        if k == 13:
            break

    cv2.destroyAllWindows()
    return (click[1] - y_min, click[0] - x_min)
