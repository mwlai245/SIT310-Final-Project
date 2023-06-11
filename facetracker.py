import cv2

import numpy as np

from djitellopy import tello

import time

me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()
# me.send_command_without_return()


w, h = 540, 360
MAX_STATE = 5

fbRange = [15 * 1000, 20 * 1000]

# up down variable
udRange = [(h / 2) - 30, (h / 2)]  # MIN DOWN, MAX UP
udMax = [(h / 2), (h / 2) + 30]
udMin = [(h / 2) - 30, (h / 2) - 60]
udMotion = [0, -10, 10, -30, 30]  # neutral, min, max, MIN , MAX

# left right variable
lrRange = [(w / 2) - 30, (w / 2) + 30]  # MIN DOWN, MAX UP
lrMax = [(w / 2) + 30, (w / 2) + 60]
lrMin = [(w / 2) - 30, (w / 2) - 60]
lrMotion = [0, 8, -8, 16, -16]

# pid
pid = [0.5, 0.5, 0]

pError = 0


def findFace(img):
    faceCascade = cv2.CascadeClassifier(
        "Resources/haarcascade_frontalface_default.xml")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 5)

    myFaceListC = []

    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cx = x + w // 2

        cy = y + h // 2

        area = w * h

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        myFaceListC.append([cx, cy])

        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:

        i = myFaceListArea.index(max(myFaceListArea))

        return img, [myFaceListC[i], myFaceListArea[i]]

    else:

        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    global x, y, area

    area = info[1]

    x, y = info[0]

    fb, ud, lr = 0, 0, 0

    error = x - w // 2

    speed = pid[0] * error + pid[1] * (error - pError)

    speed = int(np.clip(speed, -100, 100))

    udState = [y > udRange[0] and y < udRange[1], y >= udMax[0], y <= udMin[0] and area != 0, y >= udMax[1],
               y <= udMin[1] and area != 0]
    lrState = [x > lrRange[0] and x < lrRange[1], x >= lrMax[0], x <= lrMin[0] and area != 0, x >= lrMax[1],
               x <= lrMin[1] and area != 0]

    if area > fbRange[0] and area < fbRange[1]:

        fb = 0

    elif area >= fbRange[1]:

        fb = -20

    elif area <= fbRange[0] and area != 0:

        fb = 20

    for index in range(MAX_STATE):  # up down
        if (udState[index]):
            ud = udMotion[index]

    for index in range(MAX_STATE):  # left right
        if (lrState[index]):
            lr = lrMotion[index]

    ######################################
    if x == 0:
        speed = 0

        error = 0

    # print(speed, fb)

    me.send_rc_control(lr, fb, ud, speed)

    return error


# cap = cv2.VideoCapture(1)

me.takeoff()
me.send_rc_control(0, 0, 15, 0)
time.sleep(1.5)

while True:

    # _, img = cap.read()

    img = me.get_frame_read().frame

    img = cv2.resize(img, (w, h))

    img, info = findFace(img)

    pError = trackFace(info, w, pid, pError)

    # print("Center", info[0], "Area", info[1])

    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        me.streamoff()

        break
cv2.destroyAllWindows()
exit()