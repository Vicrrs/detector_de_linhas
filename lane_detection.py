import cv2
import numpy as np


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny_image = cv2.Canny(gray_image, 100, 120)

    return canny_image

# várias imagens de quadros mostradas uma após a outra
video = cv2.VideoCapture('test_video.mp4')

while video.isOpened():
    is_grabbed, frame = video.read()

    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)

    cv2.imshow('Detector de linha', frame)
    cv2.waitKey(15)

video.release()
cv2.destroyAllWindows()