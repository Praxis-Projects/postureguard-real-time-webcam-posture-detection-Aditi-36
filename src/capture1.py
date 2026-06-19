
import cv2
from matplotlib import image
import mediapipe as mp
import numpy as np 

#mediapipe components for drawing and pose detection 
mp_drawing= mp.solutions.drawing_utils 
mp_pose= mp.solutions.pose 
vid = cv2.VideoCapture(0)
# model complexity: 0 = Lite, 1 = Full, 2 = Heavy
pose= mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0 )
while vid.isOpened():
    res,frame = vid.read()
    #mediapipe reads image in bgr format, so we need to convert it to rgb for processing 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    
    #make detection
    results = pose.process(frame_rgb)
    
    #recolouring the image back to bgr for rendering 
    image.flags.writeable = True
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
   
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("video" , frame_bgr)
    

    if cv2.waitKey(10) & 0xff==ord('q'):
        break 
vid.release()
cv2.destroyAllWindows()
