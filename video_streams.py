import cv2


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


def video_stream(video_input_file, video_output_file, handler, fourcc='MP4V'):
    """
    Стрим из файла и пишет в файл, обрабатывая каждый кадр
    :param video_input_file: путь к входному видео
    :param video_output_file: путь к результату видео
    :param handler: обработчик приминает кадр с камеры и возвращает обработнный кадр
    :param fourcc: код кодека записи видео
    :return:
    """
    input_movie = cv2.VideoCapture(video_input_file)
    if not input_movie.isOpened():
        print('could not open input file')
        return

    fps = input_movie.get(cv2.CAP_PROP_FPS)
    width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    out_movie = cv2.VideoWriter(video_output_file,
                                cv2.VideoWriter_fourcc(*fourcc),
                                fps, (width, height))

    print(f'Input frame count: {count} ({width} x {height}) @ {fps:.2f} fps')

    for _ in range(count):
        if not input_movie.isOpened():
            break

        ret, frame = input_movie.read()
        if ret:
            out_movie.write(handler(frame))
        print('.', end='')

    input_movie.release()
    out_movie.release()

    print('\nDone!')
