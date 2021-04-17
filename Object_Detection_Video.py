#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/13 16:39
"""
from YOLODetection import yolo_detection
import cv2

print("[INFO]Load YOLO Model....")
modelpath = ["DarknetModels/cfg/yolov4-tiny.cfg","DarknetModels/weights/yolov4-tiny.weights"]
classespath = "ClassesFiles/coco80.names"
yolo_model = yolo_detection(modelpath,classespath,conf_threshold=0.4,nms_threshold=0.3)
print("[INFO]Load Successly!")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        res = yolo_model.predict(frame)
        frame = yolo_model.visualize(frame,res)
        cv2.imshow("YOLO",frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


