import cv2
import numpy as np


def rotate_image(mat, angle):
    """
    Вращает изображение mat на угол angle в градусах
    При это размер нового изображения подгоняется так, чтобы не было обрезки
    """

    height, width = mat.shape[:2]

    # центр вращение
    image_center = (width / 2, height / 2)

    # матрица вращения
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # косиунус угла и синус угла достанем из матрицы
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # вращаем самый дальний угол, тчобы наайт новые размеры изображения
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # заменим центр смещения в матрице вращение со старого на новый
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # применем преобразование
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rotation_mat


def camera_stream(handler, capture_size=(320, 200), capture_id=0):
    """
    Запускает стрим с камеры
    :param handler: обработчик приминает кадр с камеры и возвращает обработнный кадр
    :param capture_size: желаемый размер кадра
    :param capture_id: идентификатор камеры, если камера одна - 0
    :return: None
    """
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
    """
    Эта функция изменяет размер изображения сохраняя соотношение сторон
    А еще она одновременно трансформирует точку anchor_px = (x, y) с учетом изменения размера
    :param image: входное изобржение
    :param width: желаемая ширина или None
    :param height: желаемая высота или None
    :param inter: тип интерполяции
    :param anchor_px: точка для трасформирования или None
    :return:
    """
    (h, w) = image.shape[:2]

    # если ни одна не задана, то возвращем исходное изображение
    if width is None and height is None:
        return image

    if width is None:
        # желаем высоту, считаем ширину
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # желаем ширину, считаем высоту
        r = width / float(w)
        dim = (width, int(h * r))

    if anchor_px is not None:
        ax, ay = anchor_px
        anchor_px = (int(ax * r), int(ay * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized if anchor_px is None else (resized, anchor_px)


def rect_intersection(a, b):
    """
    Пересечение прямоугольников
    :param a: (aX, aY, aWidth, aHeight)
    :param b: (bX, bY, bWidth, bHeight)
    :return: (iX, iY, iWidth, iHeight)
    """
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
