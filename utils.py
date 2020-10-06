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


def draw_tracker_objects_and_rects(frame, objects, rects, title, titleCoordinates, idColor):
    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)

    for id, (cx, cy) in objects.items():

        text = f"L{id}"
        cv2.putText(frame, text, (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, idColor, 2)
        cv2.circle(frame, (cx, cy), 4, idColor, -1)

    cv2.putText(frame, title, titleCoordinates,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 0), 2)
