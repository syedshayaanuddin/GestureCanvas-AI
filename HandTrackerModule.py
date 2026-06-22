import cv2
import mediapipe as mp
import math
import time


class handDetector:
    # Landmark IDs for fingertips and their knuckles
    TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips

    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode         = mode
        self.maxHands     = maxHands
        self.detectionCon = detectionCon
        self.trackCon     = trackCon

        # ── Init MediaPipe (THIS was missing in the original) ──
        self.mpHands = mp.solutions.hands
        self.mpDraw  = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles

        self.hands = self.mpHands.Hands(
            static_image_mode        = self.mode,
            max_num_hands            = self.maxHands,
            min_detection_confidence = self.detectionCon,
            min_tracking_confidence  = self.trackCon,
        )

        self.lmList   = []
        self.bbox     = ()
        self.results  = None
        self.handType = "Unknown"   # "Left" or "Right"

    # ──────────────────────────────────────────────────────────
    def findHands(self, img, draw=True):
        """Detect hands in frame. Returns annotated img."""
        imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False          # minor perf boost
        self.results = self.hands.process(imgRGB)
        imgRGB.flags.writeable = True

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpStyle.get_default_hand_landmarks_style(),
                        self.mpStyle.get_default_hand_connections_style(),
                    )
        return img

    # ──────────────────────────────────────────────────────────
    def findPosition(self, img, handNo=0, draw=False):

        self.lmList = []
        self.bbox   = ()
        xList, yList = [], []

        if self.results and self.results.multi_hand_landmarks:
            if handNo >= len(self.results.multi_hand_landmarks):
                return self.lmList, self.bbox

            # hand type (mediapipe labels are mirrored for selfie cam)
            if self.results.multi_handedness:
                self.handType = self.results.multi_handedness[handNo].classification[0].label

            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            self.bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(img,
                              (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, self.bbox

    # ──────────────────────────────────────────────────────────
    def fingersUp(self):
        """
        Returns list of 5 ints [thumb, index, middle, ring, pinky].
        1 = finger up, 0 = finger down.
        Handles left/right hand for thumb direction.
        """
        if len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb — compare tip x to IP joint x (flipped for left hand)
        if self.handType == "Right":
            fingers.append(1 if self.lmList[4][1] < self.lmList[3][1] else 0)
        else:
            fingers.append(1 if self.lmList[4][1] > self.lmList[3][1] else 0)

        # Index → Pinky — tip y < PIP joint y means finger is up
        for tip in self.TIP_IDS[1:]:
            fingers.append(1 if self.lmList[tip][2] < self.lmList[tip - 2][2] else 0)

        return fingers

    # ──────────────────────────────────────────────────────────
    def findDistance(self, p1, p2, img=None, draw=True, r=12, t=2):
        """
        Pixel distance between two landmarks.
        Returns (length, img, [x1,y1,x2,y2,cx,cy]).
        Pass img=None to skip drawing.
        """
        if len(self.lmList) < 21:
            return 0, img, []

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255),   cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]

    # ──────────────────────────────────────────────────────────
    def findAngle(self, p1, p2, p3):
        """
        Angle (degrees) at landmark p2 formed by p1-p2-p3.
        Useful for gesture recognition beyond simple finger-up checks.
        """
        if len(self.lmList) < 21:
            return 0

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        return abs(angle) % 360


if __name__ == "__main__":
    cap      = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1, detectionCon=0.7)
    pTime    = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        img            = detector.findHands(img)
        lmList, bbox   = detector.findPosition(img)

        if len(lmList) == 21:
            fingers = detector.fingersUp()
            cv2.putText(img, f"Fingers: {fingers}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # pinch distance (thumb tip=4, index tip=8)
            length, img, _ = detector.findDistance(4, 8, img, draw=True)
            cv2.putText(img, f"Pinch: {int(length)}px", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cTime = time.time()
        fps   = 1 / (cTime - pTime + 1e-9)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("HandTracker V2", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
