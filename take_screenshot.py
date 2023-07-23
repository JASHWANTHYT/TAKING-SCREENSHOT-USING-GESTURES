import numpy as np
import pyautogui
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # array to hold true or false if finger is folded
            finger_fold_status = []
            for tip in finger_tips:
                # getting the landmark tip position and drawing blue circle
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                # writing condition to check if finger is folded
                # i.e checking if finger tip starting value is smaller than finger starting position which is inner landmark. for index finger
                # if finger folded changing color to green
                if tip >= 3 and lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            # checking if all fingers are folded
            if all(finger_fold_status):
                # Capture screenshot using pyautogui
                img_pil = pyautogui.screenshot()

                # Convert PIL image to OpenCV-compatible numpy array
                img_np = np.array(img_pil)

                # Convert RGB to BGR (OpenCV format)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Save the screenshot directly to the disk
                cv2.imwrite("screenshot.png", img_np)

                # Load the saved screenshot in OpenCV format
                saved_img = cv2.imread("screenshot.png")

                # Display the saved screenshot in a new window
                cv2.imshow("Saved Screenshot", saved_img)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)

    # Break out of the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
