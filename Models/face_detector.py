import cv2
import os

class FaceDetector:
    def __init__(self):
        self.input_size=(640,640)
        self.confidence_threshold=0.4
        self.nms_threshold=0.3
        self.network=cv2.FaceDetectorYN.create(os.path.dirname(__file__) + '/weights/face_detection_yunet.onnx',"",self.input_size,self.confidence_threshold,self.nms_threshold)
        
    def predict(self,image):
        img_W = int(image.shape[1])
        img_H = int(image.shape[0])

        self.network.setInputSize((img_W, img_H))
        detections = self.network.detect(image)

        return detections