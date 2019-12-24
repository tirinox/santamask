import cv2
import face_recognition
import numpy as np
from helpers import *
from collections import namedtuple


SpriteAnchored = namedtuple('SpriteAnchored', ('sprite', 'anchor'))



def apply_new_year(canvas, hat, beard, face):

    print(list(face.keys()))

    for feature in face.values():
        cv2.polylines(canvas, np.array([feature]), False, (255, 255, 255, 255), thickness=2)

    overlay_scaled_rotated(canvas, hat.sprite, (200, 50), angle=10, target_width=100, anchor_px=(314, 285))


def main():
    hat = SpriteAnchored(cv2.imread('data/hat.png', cv2.IMREAD_UNCHANGED), (314, 285))
    beard = SpriteAnchored(cv2.imread('data/beard.png', cv2.IMREAD_UNCHANGED), (606, 109))

    im = cv2.imread('data/man.jpg')
    im = image_resize(im, width=800)

    face_landmarks_list = face_recognition.face_landmarks(im)

    print(f'{len(face_landmarks_list)} faces found.')

    for face in face_landmarks_list:
        apply_new_year(im, hat, beard, face)

    cv2.imshow('1', im)
    cv2.waitKey()


if __name__ == '__main__':
    main()
