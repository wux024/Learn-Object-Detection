#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/18 17:09
@Description: Pytorch Hub comes with many pre-trained models,
such as object detection models that can be used directly.
If the model you use is not in the model library, we should
try to parse the yaml file or directly implement the model used,
and load the weight file. PyTorch Hub actually does this for us.
TensorFlow is similar.
"""
import torch
import numpy as np
import cv2

def get_names(classespath):
    # COCO Object category name
    classesFile = classespath
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

def visualize(img, results):
    classes = get_names('ClassesFiles/coco80.names')
    np.random.seed(200)
    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    for result in results:
        class_id = int(result[-1])
        color = [int(c) for c in COLORS[class_id]]
        score = '%.2f' % result[-2]
        label = '%s:%s' % (classes[class_id], score)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        left, top, right, bottom = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        cv2.rectangle(img,
                      (left, top),
                      (right, bottom),
                      color, 2)
        top = max(top, label_size[1])
        # Painting category and accuracy
        cv2.rectangle(img,
                      (left, top - round(1.5 * label_size[1])),
                      (left + round(1.5 * label_size[0]), top + baseline),
                      color,
                      cv2.FILLED)
        cv2.putText(img,
                    label,
                    (left, top),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1)
    return img

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
imgpath = 'TestImage/Naxos_Taverna.jpg'
img = cv2.imread(imgpath)
# Inference
results = model(imgpath)
results = results.xyxy[0].numpy()
img = visualize(img, results)
cv2.imshow("Detection",img)
cv2.imwrite(imgpath[:-4] + '_detection_pyhub.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

