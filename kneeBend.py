from dataclasses import dataclass
import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime

#Screen Keep your bent
last_detected = datetime.now()

def calculateAngle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 



#Time
duration = 8
globalCount = 0
sign = 0

timeDuration = 0

#Count knee bend
count = 0
state = "straight"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('KneeBendVideo.mp4')


fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (854, 640))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# Read until video is completed
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while(cap.isOpened()):
        if globalCount == 0:
            start = time.time()
            print(start)
            globalCount+= 1

        # Capture frame-by-frame
        ret, frame = cap.read()

        

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_Knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculateAngle(left_hip, left_Knee, left_ankle)

            #print(angle)

                       # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(left_Knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

            

            #Count bends of the knee
            if angle > 140 and state == "Bend":
                #print(start_time.second, bendStart)
                bendEnd = time.time()
                totalTime = bendEnd - bendStart
                timeDuration = totalTime
                if totalTime < duration:
                    count -= 1
                    sign=1

                state = "straight"
                #if bendEnd <= duration:
                #    count = 0

            if angle < 140 and state == "straight":
                bendStart = time.time()
                print(bendStart-start)
                startSignal = 1
                sign=0
                state = "Bend"
                count += 1
                print(count)
                #print(bendStart)
                

        except:
            pass
        

        #Add Clock
        cv2.rectangle(image, (1000,0), (600,90), (0,0,0), -1)

        cv2.putText(image, str(round(timeDuration)), 
                    (620,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (128,128,128), 2, cv2.LINE_AA)
        cv2.putText(image, "s", 
                    (800,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (128,128,128), 2, cv2.LINE_AA)
        #Make the Counter Visualize
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (320,90), (0,0,255), -1)
        
        # Rep data
        cv2.putText(image, 'Bends', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(count), 
                    (15,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'Leg state', (100,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, state, 
                    (80,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        if sign == 1:
            cv2.putText(image, 'Keep your knee bent', (150,280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )              
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame',image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()