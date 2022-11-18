# App Version 1
# Necessary Imports
import cv2
from deepface import DeepFace

# Download XML file for trained frontal face data from opencv:
# https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set initial color
color=(0,0,0)

# Capture Video from Webcam
# 0 for webcam BUT can specify other video files
webcam = cv2.VideoCapture(0)

while True:
    # Read current frame
    succesful_frame_read, frame = webcam.read()

    # Convert Frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(gray)

    # try and except condition in case of any errors
    try:
        # using the analyze class from deepface and using ‘frame’ as input, setting actions to emotion
        analyze = DeepFace.analyze(frame,  enforce_detection=False)
        # print dominant emotion to console
        print(analyze['dominant_emotion'])

        # Change rectangle color based on emotion
        if(analyze['dominant_emotion']=='happy'):
            color=(0,255,0) # Green
        elif(analyze['dominant_emotion']=='neutral'):
            color= (0,205,255) # Orange
        elif(analyze['dominant_emotion']=='angry'):
            color= (0,0,255) # Red
        elif(analyze['dominant_emotion']=='fear'):
            color= (180,180,180) # Gray
        elif(analyze['dominant_emotion']=='sad'):
            color = (255,0,0)  # Blue
    except:
        print("no face")

    # Draw rectangle around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)

    cv2.imshow("Webcam Face & Emotion Detector", frame)
    key=cv2.waitKey(1)

    # Hit q key to quit out of the program
    if key==81 or key==113:
        break

print("Code Completed")
