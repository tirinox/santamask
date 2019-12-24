from collections import namedtuple

import face_recognition

from helpers import *

SpriteAnchored = namedtuple('SpriteAnchored', ('sprite', 'anchor'))


def face_width(face: dict):
    chin = face['chin']
    p1 = np.array(chin[0])
    p2 = np.array(chin[-1])
    return np.linalg.norm(p1 - p2)


def center_of_feature(f):
    nt = np.array(f)
    x, y = np.mean(nt, axis=0)
    return x, y


def nose_tip(face: dict):
    x, y = center_of_feature(face['nose_tip'])
    return int(x), int(y)


def face_angle(face: dict):
    nb = face['nose_bridge']
    dp = np.array(nb[0]) - np.array(nb[-1])
    _, y = dp / np.linalg.norm(dp)
    return np.rad2deg(np.arccos(-y))


def brow_center(face: dict):
    chin = face['chin']
    p1 = np.array(chin[0])
    p2 = np.array(chin[-1])

    x, y = center_of_feature([
        p1, p2
    ])

    return int(x), int(y)


def apply_new_year(canvas, hat, beard, face):
    # print(list(face.keys()))

    fw = face_width(face)
    beard_width = int(fw * 1.1)
    hat_width = int(fw * 1.7)

    beard_pos = nose_tip(face)
    hat_pos = brow_center(face)

    angle = face_angle(face)

    print(f'Angle: {angle:.2f}º')

    overlay_scaled_rotated(canvas, hat.sprite, hat_pos, angle=angle, target_width=hat_width, anchor_px=hat.anchor)

    overlay_scaled_rotated(canvas, beard.sprite, beard_pos, angle=angle, target_width=beard_width, anchor_px=beard.anchor)

    # for feature in face.values():
    #     cv2.polylines(canvas, np.array([feature]), False, (255, 0, 255, 255), thickness=2)


def main():
    hat = SpriteAnchored(cv2.imread('data/hat.png', cv2.IMREAD_UNCHANGED), (322, 450))
    beard = SpriteAnchored(cv2.imread('data/beard.png', cv2.IMREAD_UNCHANGED), (606, 85))

    def handler(frame):
        face_landmarks_list = face_recognition.face_landmarks(frame)
        print(f'{len(face_landmarks_list)} faces found.')
        for face in face_landmarks_list:
            apply_new_year(frame, hat, beard, face)
        return frame

    camera_stream(handler, (640, 480))

    # im = cv2.imread('data/man.jpg')
    # im = image_resize(im, width=800)
    # cv2.imshow('1', im)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
