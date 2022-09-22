import cv2
import numpy as np


def region_of_interest(image, region_points):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region_points, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny_image = cv2.Canny(gray_image, 100, 120)

    region_of_interest_vertices = [
        (0, height),
        (width/2, height*0.5),
        (width, height)
    ]

    cropped_image = region_of_interest(
        canny_image, np.array([region_of_interest_vertices], np.int32))

    return cropped_image

# link para o vídeo (https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project?resource=download)
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
