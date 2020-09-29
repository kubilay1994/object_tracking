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

parser.add_argument("-v", "--video", type=str, help="Video dosyasının yolu")
args = parser.parse_args()


if args.video:
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.video))
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened:
    print("Video açılamadı")
    exit(0)

backSub = algorithms[args.algo]()
while True:

    ret, frame = cap.read()
    if ret == False:
        break

    mask = backSub.apply(frame)

    if args.algo == "gmg":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [
        contour for contour in contours if cv2.contourArea(contour) > 500]

    # cv2.drawContours(frame, contours, -1, (100, 0, 100), -1)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)

    cv2.imshow("normal", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
