#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/13 20:31
"""
import cv2
import numpy as np
import random

class maskrcnn_detection(object):
    def __init__(self, modelpath, classespath, conf_threshold=0.6,mask_threshold=0.3,is_GPU = False):
        self.conf_threshold = conf_threshold  # Confidence threshold
        self.mask_threshold = mask_threshold  # Mask threshold
        self.maskrcnn_model = self.get_maskrcnn_net(modelpath,is_GPU)
        self.classes = self.get_classes_name(classespath)


    def get_classes_name(self,classespath):
        # Load names of classes
        classesFile = classespath
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        return classes


    def get_maskrcnn_net(self, modelpath,is_GPU):
        pb_file = modelpath[0]
        pbtxt_file = modelpath[1]
        maskrcnn_model = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
        if not is_GPU:
            maskrcnn_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            maskrcnn_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            # Here is the NVIDIA GPU, and some other GPUs are also supported
            maskrcnn_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            maskrcnn_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return maskrcnn_model


    def postprocess(self, boxes, masks, img_height, img_width):
        # For each frame, extract the bounding box and mask of each detected object
        # The output size of the mask is NxCxHxW
        # N - the number of detected boxes
        # C - Number of categories (excluding background)
        # HxW - Split shape
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]

        results = []
        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.conf_threshold:
                left = int(img_width * box[3])
                top = int(img_height * box[4])
                right = int(img_width * box[5])
                bottom = int(img_height * box[6])

                left = max(0, min(left, img_width - 1))
                top = max(0, min(top, img_height - 1))
                right = max(0, min(right, img_width - 1))
                bottom = max(0, min(bottom, img_height - 1))

                result = {}
                result["score"]  = score
                result["classid"] = int(box[1])
                result["box"]   = (left, top, right, bottom)
                result["mask"]   = mask[int(box[1])]

                results.append(result)

        return results


    def predict(self, img_cv2):
        img_height, img_width, _ = img_cv2.shape

        # Create a 4D blob from the frame
        blob = cv2.dnn.blobFromImage(img_cv2, swapRB=True, crop=True)

        # Set the input of the network
        self.maskrcnn_model.setInput(blob)

        # Run forward pass to get output from the output layer
        boxes, masks = self.maskrcnn_model.forward(
            ['detection_out_final', 'detection_masks'])

        # Extract bounding box and mask for each detected object
        results = self.postprocess(boxes, masks, img_height, img_width)

        return results


    def visualize(self, img_cv2, results):
        np.random.seed(100)
        COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        for result in results:
            color = [int(c) for c in COLORS[result["classid"]]]
            # box
            left, top, right, bottom = result["box"]
            cv2.rectangle(img_cv2,
                          (left, top),
                          (right, bottom),
                          color, 3)

            # class label
            classid = result["classid"]
            score   = result["score"]
            label = '%.2f' % score
            if self.classes:
                assert (classid < len(self.classes))
                label = '%s:%s' % (self.classes[classid], label)

            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            top = max(top, label_size[1])

            cv2.rectangle(
                img_cv2,
                (left, top - round(1.5 * label_size[1])),
                (left + round(1.5 * label_size[0]), top + baseline),
                color, cv2.FILLED)
            cv2.putText(
                img_cv2,
                label,
                (left, top),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

            # mask
            class_mask = result["mask"]
            class_mask = cv2.resize(class_mask, (right - left + 1, bottom - top + 1))
            mask = (class_mask > self.mask_threshold)
            roi = img_cv2[top: bottom + 1, left: right + 1][mask]

            np.random.seed(150)
            color_index = random.randint(0, len(self.classes) - 1)
            color = [int(c) for c in COLORS[color_index]]
            img_cv2[top: bottom + 1, left: right + 1][mask] = (
                        [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

            mask = mask.astype(np.uint8)
            contours, hierachy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                img_cv2[top: bottom + 1, left: right + 1],
                contours,
                -1,
                color,
                3,
                cv2.LINE_8,
                hierachy,
                100)

        return img_cv2

