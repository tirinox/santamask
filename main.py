from santamask import SantaMask
from helpers import camera_stream


def main():
    santa = SantaMask()

    camera_stream(santa.apply, (640, 480))

    # im = cv2.imread('data/man.jpg')
    # im = image_resize(im, width=800)
    # cv2.imshow('1', im)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
