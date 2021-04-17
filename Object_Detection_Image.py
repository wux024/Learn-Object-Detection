#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/13 8:20
@Description:
Why use OpenCV's dnn module? OpenCV's dnn module can only infer but not train.
Because of this, he has three advantages. First, light weight. Second, it is
easy to use.Third, versatility.
COCO? The COCO data set target detection is divided into 80 categories,and the
instance segmentation has 91 categories. The pre-training model officially
provided by TensorFlow is calculated according to instance segmentation.The
labeling order of SSD and RCNN series is not the same. SSD type 0 is the
background, and the 90th type of the RCNN series is the background.
"""
from OpenCVTFDetection import detection_tfmodel
from YOLODetection import yolo_detection
from MaskRCNNDetection import maskrcnn_detection
import cv2
import numpy as np

if __name__ == '__main__':
    img_file = "TestImage/dog.jpg"
    img = cv2.imread(img_file)
    # h, w, _ = img.shape
    # img = cv2.resize(img,(w//4,h//4))
    print("[INFO]YOLO")
    modelpath = ["DarknetModels/cfg/yolov4-mish-608.cfg","DarknetModels/weights/yolov4-mish-608.weights"]
    classespath = "ClassesFiles/coco80.names"
    yolo_model = yolo_detection(modelpath,classespath,net_width=608,net_height=608)
    res = yolo_model.predict(img.copy())
    img1 = yolo_model.visualize(img.copy(),res)

    print("[INFO]Faster RCNN")
    modelpath = ["TensorFlowDNNModels/pb/faster_rcnn_resnet50_coco_2018_01_28.pb",
                 "TensorFlowDNNModels/pbtxt/faster_rcnn_resnet50_coco_2018_01_28.pbtxt"]
    classespath = "ClassesFiles/coco_rcnn.names"
    fasterrcnn_tf = detection_tfmodel(modelpath,classespath, conf_threshold=0.4,nms_threshold=0.4,net_width=416,net_height=416)
    res = fasterrcnn_tf.predict(img.copy())
    img2 = fasterrcnn_tf.visualize(img.copy(),res)

    print("[INFO]SSD")
    modelpath = ["TensorFlowDNNModels/pb/ssd_mobilenet_v2_coco_2018_03_29.pb",
                 "TensorFlowDNNModels/pbtxt/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"]
    classespath = "ClassesFiles/coco_ssd.names"
    ssd_tf = detection_tfmodel(modelpath, classespath, conf_threshold=0.3,nms_threshold=0.4, net_width=300,net_height=300)
    res = ssd_tf.predict(img.copy())
    img3 = ssd_tf.visualize(img.copy(), res)

    print("[INFO]Mask RCNN")
    modelpath = ["TensorFlowDNNModels/pb/mask_rcnn_inception_v2_coco_2018_01_28.pb",
                 "TensorFlowDNNModels/pbtxt/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"]
    classespath = "ClassesFiles/coco_rcnn.names"
    maskrcnn_tf = maskrcnn_detection(modelpath, classespath, conf_threshold=0.55, mask_threshold=0.3)
    res = maskrcnn_tf.predict(img.copy())
    img4 = maskrcnn_tf.visualize(img.copy(), res)


    # cv2.imshow('Object Detection Based on YOLO',img1)
    # cv2.imshow('Object Detection Based on Faster RCNN', img2)
    # cv2.imshow('Object Detection Based on SSD', img3)
    # cv2.imshow('Object Detection Based on Mask RCNN', img4)
    cv2.putText(img1, 'YOLO',(0,40),cv2.FONT_HERSHEY_COMPLEX,0.618,(0,0,255))
    cv2.putText(img2, 'Faster RCNN', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.618, (0, 0, 255))
    cv2.putText(img3, 'SSD', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.618, (0, 0, 255))
    cv2.putText(img4, 'Mask RCNN', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.618, (0, 0, 255))
    img = np.vstack((np.hstack((img1,img2)),np.hstack((img3,img4))))
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Object Detection',img)
    cv2.imwrite(img_file[:-4] + '_detection.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


