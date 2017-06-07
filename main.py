import datetime
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while (True):

    now = datetime.datetime.now()
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)    # Command to mirror the frame

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 127, 255, 0)
    # ret, thresh = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)   # Thresbold No.2

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        max_index = np.argmax(area)
        cnt = contours[max_index]

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        for p in box:
            pt = (p[0], p[1])
            cv2.circle(frame, pt, 5, (0, 0, 255), 2)

        ''' Calculate Vectors '''

        # Tilt Left (Lower Corner Left Side)
        x1 = math.sqrt(pow(box[3, 0] - box[0, 0], 2) + pow(box[3, 1] - box[0, 1], 2))
        y1 = math.sqrt(pow(box[0, 0] - box[1, 0], 2) + pow(box[0, 1] - box[1, 1], 2))

        # Tilt Right (Lower Corner Right Side)
        x2 = math.sqrt(pow(box[0, 0] - box[1, 0], 2) + pow(box[0, 1] - box[1, 1], 2))
        y2 = math.sqrt(pow(box[0, 0] - box[3, 0], 2) + pow(box[0, 1] - box[3, 1], 2))

        ''' End of Vector Calculation '''

        if x1 > x2:
            x = x1
            y = y1
        else:
            x = x2
            y = y2

        # Scale for my specific setup (distance from object)
        x = 0.48 * x
        y = 0.48 * y

        ''' Text Output on Screen '''

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Show Coordinates for each corner
        cv2.putText(frame, ('Corner 1: x=%d  y=%d' % (box[0, 0], box[0, 1])), (5, 20), font, 0.7, (255, 127, 127), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, ('Corner 2: x=%d  y=%d' % (box[1, 0], box[1, 1])), (5, 40), font, 0.7, (255, 127, 127), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, ('Corner 3: x=%d  y=%d' % (box[2, 0], box[2, 1])), (5, 60), font, 0.7, (255, 127, 127), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, ('Corner 4: x=%d  y=%d' % (box[3, 0], box[3, 1])), (5, 80), font, 0.7, (255, 127, 127), 1,
                    cv2.LINE_AA)
        # Show current date and time
        cv2.putText(frame, (str(now)[:19]), (480, 20), font, 0.4, (255, 127, 127), 1, cv2.LINE_AA)
        # Show the dimensions of the shape
        cv2.putText(frame, ('Dimension = %d x %d mm' % (x, y)), (5, 465), font, 1, (255, 127, 127), 1, cv2.LINE_AA)

        ''' End of Text Output on Screen '''

    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
