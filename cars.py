import cv2
import numpy as np
from centroidTracker import CentroidTracker
from utils import register_tracker_to_roi, draw_tracker_objects_and_rects

cap = cv2.VideoCapture("videos/2.mp4")

backSub = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter(f'videos/output4.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

ct1 = CentroidTracker(maxDissappeared=15)
ct2 = CentroidTracker(maxDissappeared=15)

minArea = 500
minWidth = 12
minHeight = 35

l_id = r_id = 0


for i in range(50):
    ret, frame = cap.read()
    if ret == False:
        break
    backSub.apply(frame)

while True:

    ret, frame = cap.read()

    if ret == False:
        break

    H, W, _ = frame.shape
    W = W + 100
    cv2.line(frame, (W//2, 0), (W//2, H), (0, 0, 0), 2)

    mask = backSub.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_TOZERO)

    lobjects, lrects = register_tracker_to_roi(
        roi=mask[:, :W//2], start=0, tracker=ct1, minArea=minArea, minHeight=minHeight, minWidth=minWidth)

    robjects, rrects = register_tracker_to_roi(
        roi=mask[:, W//2:W], start=W//2, tracker=ct2, minArea=minArea, minHeight=minHeight, minWidth=minWidth)

    l_id = ct1.get_count()
    r_id = ct2.get_count()

    draw_tracker_objects_and_rects(
        frame, lobjects, lrects,
        title=f"Kuzeye dogru giden nesne sayisi : {l_id}", titleCoordinates=(0, 50), idColor=(0, 0, 255))

    draw_tracker_objects_and_rects(
        frame, robjects, rrects,
        title=f"Guneye dogru giden nesne sayisi : {r_id}", titleCoordinates=(W//2 + 50, 50), idColor=(0, 255, 0))

    cv2.imshow("normal", frame)
    cv2.imshow("mask", mask)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
