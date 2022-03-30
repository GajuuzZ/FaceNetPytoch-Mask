import cv2
import numpy as np


def resize_padding(image, width, height, pad_mode=cv2.BORDER_CONSTANT, pad_value=0,
                   interpolation=None, pad_to='center'):
    assert pad_to in ['center', 'lefttop'], 'Invalid pad_to!!'
    o_size = image.shape[:2]
    d_size = (height, width)

    idx = np.argmax(o_size)
    ratio = float(d_size[idx]) / max(o_size)
    n_size = tuple([int(np.round(x * ratio)) for x in o_size])
    if n_size > d_size:
        idx = int(not idx)
        ratio = float(d_size[idx]) / min(o_size)
        n_size = tuple([int(np.round(x * ratio)) for x in o_size])

    target_h, target_w = n_size
    if interpolation is None:
        interpolation = cv2.INTER_AREA if o_size[0] * o_size[1] > target_h * target_w \
            else cv2.INTER_LINEAR
    image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    if pad_to == 'lefttop':
        pad_w = width - target_w
        pad_h = height - target_h
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, pad_mode, value=pad_value)
    else:
        pad_w = (width - target_w) // 2
        pad_h = (height - target_h) // 2
        image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, pad_mode, value=pad_value)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image, ratio, (pad_w, pad_h)
