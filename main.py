from santamask import SantaMask
from video_streams import camera_stream, video_stream
import sys


def main():
    santa = SantaMask()
    handler = santa.apply

    if len(sys.argv) == 3:
        # если заданы аргументы - то читаем и пишем в видео
        _, input_videofile, output_videofile = sys.argv
        video_stream(input_videofile, output_videofile, handler)
    else:
        # без аргументов - видео будет с камеры
        camera_stream(handler, (640, 480))


if __name__ == '__main__':
    main()
