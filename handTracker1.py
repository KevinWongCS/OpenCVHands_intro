import cv2
# mediapipe has alot of resources with predefined solutions
import mediapipe as mp
import time

# Define a video capture object
# "0" is for webcam #1
vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

# hold ctrl and left click to see the methods definition
# this specific method has default parameters
# note: this method only uses RGB
hands = mpHands.Hands(False)

# superimposes or draws solutions for "hands" on the RGB video
# must convert BGR to RGB
mpDraw = mp.solutions.drawing_utils

# initializing pTime and cTime
pTime = 0
cTime = 0

while True:
    # Capture the video frame by frame
    ret, img = vid.read()
    # have to convert "img"(BGR image) to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # printing the id, x and y coordinates to determine the position
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # we are using int() to convert the decimals to integers everytime int() is used
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                # drawing a bigger circle for base id, id == 0 is base and id == 4 is a tip of thumb
                # if you remove the "if" part, it draws the circles for all of them
                if id == 0:
                    cv2.circle(img, (cx, cy), 35, (205, 0, 235), cv2.FILLED)

            # note: we are displaying BGR image, but the solution needs an RGB solution
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    # Below displays frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the resulting frame
    cv2.imshow('Image', img)

    # waitKey(0) will pause your screen because it will wait infinitely for keyPress on your keyboard
    # and will not refresh the frame(cap.read()) using your WebCam. waitKey(1) will wait for keyPress
    # for just 1 millisecond and it will continue to refresh and read frame from your webcam using cap.read().
    cv2.waitKey(1)

