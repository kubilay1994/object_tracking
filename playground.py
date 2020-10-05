import cv2
import numpy as np

import argparse

algorithms = dict(
    mog=cv2.bgsegm.createBackgroundSubtractorMOG,
    mog2=cv2.createBackgroundSubtractorMOG2,
    knn=cv2.createBackgroundSubtractorKNN,
    gmg=cv2.bgsegm.createBackgroundSubtractorGMG
)

parser = argparse.ArgumentParser(
    description="Arkaplan çıkarma algoritmaları ile nesne takibi")

parser.add_argument("-a", "--algo", help="Arkaplan algoritması", type=str,
                    choices=algorithms.keys(), default="mog2")

# parser.add_argument("-v", "--video", type=str, help="Video dosyasının yolu")
args = parser.parse_args()

cap = cv2.VideoCapture(0)

frame_num = 0
backSub = algorithms[args.algo]()

while True:

    ret, frame = cap.read()

    if ret == False:
        break

    rects = []
    mask = backSub.apply(frame)

    if args.algo == "gmg":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.dilate(mask, kernel)
        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [
        contour for contour in contours if cv2.contourArea(contour) > 1000]

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        rects.append((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)

    frame_num += 1

    cv2.imshow("normal", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
