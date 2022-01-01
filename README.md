# OpenCVHands_intro
name: kevin wong\
date: 1/1/21\
file: OpenCVHands_intro\
desc: openCV handtracking1 from [tutorial](https://www.youtube.com/watch?v=01sAkU_NvOY) 

1. Import cv2, mediapipe, time
```
import cv2
import mediapipe as mp  # mediapipe has alot of resources with predefined solutions
import time
```

2. Define VideoCapture object
```
vid = cv2.VideoCapture(0)
```

3. Define mpHands object 
```
hands = mpHands.Hands(False)
``` 
note: mp = media pipe

4. Define draw utility
```
mpDraw = mp.solutions.drawing_utils
```

5. Initialize present time and current time
```
pTime = 0
cTime = 0
```

6. Capture video by  frame
```
while True:
ret, img = vid.read()
```
note: steps 7 and 8 are nested in number 6's "while True loop"

7. Convert BGR to RGB because you can only "draw" on RGB images
```
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)
print(results.multi_hand_landmarks)
mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
```
note: "mpDraw.draw_landmarks" draws the connecting lines between the landmarks(dots)

8. printing the id, x and y coordinates to determine the position
```
for id, lm in enumerate(handLms.landmark):
    h, w, c = img.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
    print(id, cx, cy)

    if id == 0:
        cv2.circle(img, (cx, cy), 35, (205, 0, 235), cv2.FILLED)
```
note: we are using int() to convert the decimals to integers everytime int() is used
note: drawing a bigger circle for base id, id == 0 is base and id == 4 is a tip of thumb
note: removing the "if" statement draws the circles settings for all landmarks

9. Displaying framerate
```
cTime = time.time()
fps = 1/(cTime - pTime)
pTime = cTime
cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
```

10. Display the resulting frame
```
cv2.imshow('Image', img)
```

11. Set frame so that it stays in view
```
cv2.waitKey(1)
```
note: waitKey(0) will pause your screen because it will wait infinitely for keyPress on your keyboard and will not refresh the frame(cap.read()) using your WebCam. waitKey(1) will wait for keyPress for just 1 millisecond and it will continue to refresh and read frame from your webcam using cap.read().
