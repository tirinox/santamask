from collections import namedtuple

import face_recognition

from helpers import *

# просто тип данных, чтобы хранить картинку и ее якорь
SpriteAnchored = namedtuple('SpriteAnchored', ('sprite', 'anchor'))


def face_width(face: dict):
    """
    Ширина лица
    :param face: лицо
    :return: float
    """
    chin = face['chin']

    # расстояние между крайними точками подбородка
    p1 = np.array(chin[0])
    p2 = np.array(chin[-1])
    return np.linalg.norm(p1 - p2)


def center_of_feature(f):
    """
    Средняя среди точек
    :param f: List[Tuple(x, y)] список точке
    :return:
    """
    nt = np.array(f)
    x, y = np.mean(nt, axis=0)
    return x, y


def nose_tip(face: dict):
    """
    Кончик носа
    :param face: лицо
    :return: (x, y)
    """
    x, y = center_of_feature(face['nose_tip'])
    return int(x), int(y)


def face_angle(face: dict):
    """
    Угол наклона лица
    :param face: лицо
    :return: float - угол в градусах
    """
    nb = face['nose_bridge']   # линия переносицы
    dp = np.array(nb[0]) - np.array(nb[-1])  # вектор вдоль этой линии

    x, y = dp / np.linalg.norm(dp)  # нормируем его
    return np.rad2deg(np.arctan2(x, y) - np.pi)


def brow_center(face: dict):
    """
    Центр лба
    :param face: лицо
    :return:
    """

    # крайние точки линии подбородка
    chin = face['chin']
    p1 = np.array(chin[0])
    p2 = np.array(chin[-1])

    # центр между этими линиями
    x, y = center_of_feature([
        p1, p2
    ])

    return int(x), int(y)


class SantaMask:
    def __init__(self, debug=False):
        # шапка
        self.hat = SpriteAnchored(cv2.imread('data/hat.png', cv2.IMREAD_UNCHANGED), (322, 450))
        # борода
        self.beard = SpriteAnchored(cv2.imread('data/beard.png', cv2.IMREAD_UNCHANGED), (606, 85))
        self.debug = debug

    def _apply_for_face(self, canvas, face):
        fw = face_width(face)  # ширина лица
        beard_width = int(fw * 1.1)  # ширина бороды пошире
        hat_width = int(fw * 1.7)  # ширина шапки еще шире

        # борода крепится к кончику носа
        beard_pos = nose_tip(face)

        # шап крепится ко лбу
        hat_pos = brow_center(face)

        # угол наклона лица от вертикали
        angle = face_angle(face)

        # рисуем шапку
        overlay_scaled_rotated(canvas,
                               self.hat.sprite,
                               hat_pos, angle=angle,
                               target_width=hat_width,
                               anchor_px=self.hat.anchor)

        # рисуем бороду
        overlay_scaled_rotated(canvas,
                               self.beard.sprite,
                               beard_pos, angle=angle,
                               target_width=beard_width,
                               anchor_px=self.beard.anchor)

        # в случае отладки
        if self.debug:
            print(f'Angle: {angle:.2f}º')
            for feature in face.values():
                cv2.polylines(canvas, np.array([feature]), False, (255, 0, 255, 255), thickness=2)

    def apply(self, frame):
        # находим все лица
        face_landmarks_list = face_recognition.face_landmarks(frame)

        if self.debug:
            print(f'{len(face_landmarks_list)} faces found.')

        # для каждого лица - рисуем шапку и бороду
        for face in face_landmarks_list:
            self._apply_for_face(frame, face)
        return frame
