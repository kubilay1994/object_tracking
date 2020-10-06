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

parser.add_argument("-i", "--input", help="Video input", default=None)
parser.add_argument("-bw", "--boxwidth",
                    help="Minimum genişlik", default=None, type=int)
parser.add_argument("-bh", "--boxheight",
                    help="Minimum yükseklik", default=None, type=int)
parser.add_argument("-area", "--boxarea", default=None, type=int)


args = parser.parse_args()


minWidth = args.boxwidth if args.boxwidth else 40
minHeight = args.boxheight if args.boxwidth else 40
minArea = minHeight * minWidth * 3 / 4 if not args.boxarea else 1200

cap = cv2.VideoCapture(args.input) if args.input else cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter(f'videos/{args.out}.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

ct = CentroidTracker(maxDissappeared=15)
backSub = algorithms[args.algo]()


for i in range(100):
    ret, frame = cap.read()
    if ret == False:
        break
    backSub.apply(frame)

max_id = -1
while True:

    ret, frame = cap.read()

    if ret == False:
        break

    rects = []
    mask = backSub.apply(frame)

    if args.algo == "gmg":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.erode(mask, kernel)
    elif args.algo == "mog2":
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_TOZERO)

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > minArea and w > minWidth and h > minHeight:
            rects.append((x, y, w, h))
            rects.append((x, y, w, h))

    rects, _ = cv2.groupRectangles(rects, 1, 0.2)

    objects = ct.update(rects)

    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)

    for id, (cx, cy) in objects.items():

        max_id = max(max_id, id)
        text = f"{id}"
        cv2.putText(frame, text, (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.putText(frame, f"Sahneye giren nesne sayisi : {max_id + 1}", (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 0), 2)

    cv2.imshow("normal", frame)
    cv2.imshow("mask", mask)

    if args.out:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
