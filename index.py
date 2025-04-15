import numpy as np
import dlib
import cv2

from math import hypot

cap = cv2.VideoCapture(0) # capture the frames from the webcam in an infinite loop till we break it and stop the capture

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Calculating the blinking ratio or the eye aspect ratio of the eyes
# Starting from the left corner moving clockwise. 
# We find the ratio of height and width of the eye to infer the open or close state of the eye.
# blink-ratio=(|p2-p6|+|p3-p5|)(2|p1-p4|). 
# The ratio falls to approximately zero when the eye is close but remains constant when they are open.
def mid(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def eye_aspect_ratio(eye_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(eye_landmark[0]).x, face_roi_landmark.part(eye_landmark[0]).y)
    right_point = (face_roi_landmark.part(eye_landmark[3]).x, face_roi_landmark.part(eye_landmark[3]).y)

    center_top = mid(face_roi_landmark.part(eye_landmark[1]), face_roi_landmark.part(eye_landmark[2]))
    center_bottom = mid(face_roi_landmark.part(eye_landmark[5]), face_roi_landmark.part(eye_landmark[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

# We define the mouth ratio function for finding out if a person is yawning or not. 
# This function gives the ratio of height to width of mouth. 
# If height is more than width it means that the mouth is wide open.
def mouth_aspect_ratio(lips_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(lips_landmark[0]).x, face_roi_landmark.part(lips_landmark[0]).y)
    right_point = (face_roi_landmark.part(lips_landmark[2]).x, face_roi_landmark.part(lips_landmark[2]).y)

    center_top = (face_roi_landmark.part(lips_landmark[1]).x, face_roi_landmark.part(lips_landmark[1]).y)
    center_bottom = (face_roi_landmark.part(lips_landmark[3]).x, face_roi_landmark.part(lips_landmark[3]).y)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if hor_line_length == 0:
        return ver_line_length
    ratio = ver_line_length / hor_line_length
    return ratio

# We create a counter variable to count the number of frames the eye has been close for or the person is yawning 
# and later use to define drowsiness in driver drowsiness detection system project
count = 0

font = cv2.FONT_HERSHEY_TRIPLEX

# We flip the frame because mirror image and convert it to grayscale. Then pass it to the face detector.
while True:
    _, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    for face_roi in faces:

        landmark_list = predictor(gray, face_roi)

        left_eye_ratio = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmark_list)
        right_eye_ratio = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmark_list)
        eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2
        cv2.putText(img, str(eye_open_ratio), (0, 13), font, 0.5, (100, 100, 100))
        ###print(left_eye_ratio,right_eye_ratio,eye_open_ratio)

        inner_lip_ratio = mouth_aspect_ratio([60,62,64,66], landmark_list)
        outter_lip_ratio = mouth_aspect_ratio([48,51,54,57], landmark_list)
        mouth_open_ratio = (inner_lip_ratio + outter_lip_ratio) / 2;
        cv2.putText(img, str(mouth_open_ratio), (448, 13), font, 0.5, (100, 100, 100))
        ###print(inner_lip_ratio,outter_lip_ratio,mouth_open_ratio)

# Now that we have our data we check if the mouth is wide open and the eyes are not closed. 
# If we find that either of these situations occurs we increment the counter variable 
# counting the number of frames the situation is persisting.
# If the eyes are close or yawning occurs for more than 10 consecutive frames
#  we infer the driver as drowsy and print that on the image as well as creating the bounding box red, 
# else just create a green bounding box
        if mouth_open_ratio > 0.380 and eye_open_ratio > 4.0 or eye_open_ratio > 4.30:
            count +=1
        else:
            count = 0
        x,y = face_roi.left(), face_roi.top()
        x1,y1 = face_roi.right(), face_roi.bottom()
        if count>10:
            cv2.rectangle(img, (x,y), (x1,y1), (0, 0, 255), 2)
            cv2.putText(img, "Sleepy", (x, y-5), font, 0.5, (0, 0, 255))
            
        else:
            cv2.rectangle(img, (x,y), (x1,y1), (0, 255, 0), 2)
            cv2.putText(img, "Not Sleepy", (x, y-5), font, 0.5, (0, 255, 0))

    cv2.imshow("img", img)

    key = cv2.waitKey(1)
    if key == 27:
        break
# Finally, we show the frame and wait for the esc keypress to exit the infinite loop.
# After we exit the loop we release the webcam capture and close all the windows and exit the program.
cap.release()

cv2.destroyAllWindows()