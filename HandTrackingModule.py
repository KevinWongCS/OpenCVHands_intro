import cv2
# mediapipe has alot of resources with predefined solutions
import mediapipe as mp
import time

class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


        # new media pipe Hands object
        self.mpHands = mp.solutions.hands
        # note: this method only uses RGB
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)

        # superimposes or draws solutions for "hands" on the RGB video
        # must convert BGR to RGB

        # this object draws lines between landmarks
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        # We have to convert "img"(BGR image) to RGB for the media pipe to work
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # note: we are displaying BGR image, but the solution needs an RGB solution
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        # image(img) with drawing on top
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # landmark list this function will return
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # printing the id, x and y coordinates to determine the position
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # we are using int() to convert the decimals to integers everytime int() is used
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                # drawing a bigger circle for base id, id == 0 is base and id == 4 is a tip of thumb
                # if you remove the "if" part, it draws the circles for all of them
                if draw:
                    cv2.circle(img, (cx, cy), 15, (205, 0, 235), cv2.FILLED)

        return lmList


def main():
    # initializing pTime and cTime
    pTime = 0
    cTime = 0

    # Define a video capture object
    # "0" is for webcam #1
    vid = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        # Capture the video frame by frame
        ret, img = vid.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        # print position of any landmark
        if len(lmList) != 0:
            print(lmList[4])

        # Below displays frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('Image', img)

        # waitKey(0) will pause your screen because it will wait infinitely for keyPress on your keyboard
        # and will not refresh the frame(cap.read()) using your WebCam. waitKey(1) will wait for keyPress
        # for just 1 millisecond and it will continue to refresh and read frame from your webcam using cap.read().
        cv2.waitKey(1)

if __name__ == "__main__":
    main()