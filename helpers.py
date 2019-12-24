import cv2
import numpy as np


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rotation_mat


def camera_stream(handler, capture_size=(320, 200), capture_id=0):
    webcam = cv2.VideoCapture(capture_id)
    print(f'Capture {capture_id} started @ {capture_size}!')

    while True:
        (_, input_frame) = webcam.read()

        input_frame = cv2.resize(input_frame, capture_size)
        output_frame = handler(input_frame)

        cv2.imshow('Output', output_frame)
        key = cv2.waitKey(10)
        if key == 27:
            break


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA, anchor_px=None):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    if anchor_px is not None:
        ax, ay = anchor_px
        anchor_px = (int(ax * r), int(ay * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized if anchor_px is None else (resized, anchor_px)


def rect_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return None
    return x, y, w, h


def overlay(dest_img, source_img, position):
    x, y = position
    x = int(x)
    y = int(y)
    sh, sw = source_img.shape[:2]
    dh, dw = dest_img.shape[:2]

    ix, iy, iw, ih = rect_intersection((0, 0, dw, dh), (x, y, sw, sh))
    sx, sy, sw, sh = rect_intersection((0, 0, sw, sh), (-x, -y, dw, dh))

    if iw > 0 and ih > 0:
        alpha_s = source_img[sy:(sy + sh), sx:(sx + sw), 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            dest_img[iy:(iy + ih), ix:(ix + iw), c] = (alpha_s * source_img[sy:(sy + sh), sx:(sx + sw), c] +
                                             alpha_l * dest_img[iy:(iy + ih), ix:(ix + iw), c])


def overlay_scaled_rotated(dest, source, position, angle, target_width, anchor_px=(0, 0)):
    scaled, anchor_px = image_resize(source, width=target_width, anchor_px=anchor_px)

    rotated, rot_transform = rotate_image(scaled, angle)

    opx, opy = anchor_px
    rop = rot_transform.dot(np.array((opx, opy, 1)))
    ropx, ropy = rop

    x, y = position
    overlay(dest, rotated, (x - ropx, y - ropy))
