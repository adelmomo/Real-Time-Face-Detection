from Models import FaceDetector
from Utils import Utils
import argparse
import cv2
import time
args_parser=argparse.ArgumentParser()
args_parser.add_argument("-i","--camera_source",type=int,default=0,help="The Camera Source")
args_parser.add_argument("-b","--face_bluring",action="store_true",help="Choose whether to blur faces in the scene or not")
args=vars(args_parser.parse_args())

face_detector=FaceDetector()
cap=Utils.create_video_capture(args['camera_source'])

while True:
    ret,frame=cap.read()
    if ret:

        inference_start_time=time.time()
        #detect faces in the image scene
        detections=face_detector.predict(frame)
        inference_end_time=time.time()
        fps=1/(inference_end_time-inference_start_time)
        formatted_fps = "{:.2f}".format(fps)
        #blur faces detected in the image scene if that is required by the user
        if args['face_bluring']:
            frame=Utils.face_bluring(frame,detections)
        else:
            frame=Utils.draw_detections(frame,detections)
        Utils.write_string_to_frame(frame,"Press Q to exit the main program",(10,20),font_scale=0.5,thickness=1,color=(0,0,255))
        Utils.write_string_to_frame(frame,f"FPS: {formatted_fps}",(10,50),font_scale=0.5,thickness=1,color=(0,0,255))
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
        cv2.imshow('Demo',frame)
        
    else:
        break