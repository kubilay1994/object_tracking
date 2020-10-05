import cv2
import numpy as np
import argparse

from centroidTracker import CentroidTracker


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

parser.add_argument("-o", "--out", help="Video kaydı", type=str, default=None)

args = parser.parse_args()


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(f'videos/{args.out}.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

ct = CentroidTracker()
backSub = algorithms[args.algo]()


for i in range(100):
    ret, frame = cap.read()
    if ret == False:
        break
    backSub.apply(frame)

max_id = 0
while True:

    ret, frame = cap.read()

    if ret == False:
        break

    rects = []
    mask = backSub.apply(frame)

    if args.algo == "gmg":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.erode(mask, kernel)

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 1200 and w > 40 and h > 40:
            rects.append((x, y, w, h))
            rects.append((x, y, w, h))

    rects, _ = cv2.groupRectangles(rects, 1, 0.5)

    objects = ct.update(rects)

    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)

    for id, (cx, cy) in objects.items():

        max_id = max(max_id, id)
        text = f"ID {id}"
        cv2.putText(frame, text, (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 200), 2)
        cv2.circle(frame, (cx, cy), 4, (80, 255, 200), -1)

    cv2.putText(frame, f"Toplam nesne sayisi : {max_id + 1}", (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 0, 0), 3)

    cv2.imshow("normal", frame)
    cv2.imshow("mask", mask)

    if args.out:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
