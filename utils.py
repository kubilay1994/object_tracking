import numpy as np
import cv2


def register_tracker_to_roi(roi, start, tracker, minArea, minWidth, minHeight):
    rects = []

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > minArea and w > minWidth and h > minHeight:
            rects.append((start + x, y, w, h))
            rects.append((start + x, y, w, h))

    rects, _ = cv2.groupRectangles(rects, 1, 0.3)

    objects = tracker.update(rects)

    return objects, rects
