#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/13 9:02
@Description: It mainly supports YOLO trained through Darknet,
but the dnn module does not support all training formats.
For example, I have failed to load YOLO-SSP.Although YOLO9000 can be loaded,
it does not work normally.
"""
import cv2
import numpy as np

class yolo_detection(object):
    def __init__(self, modelpath, classespath,conf_threshold=0.5,nms_threshold=0.4,net_width=416, net_height=416, is_GPU=False):
        self.conf_threshold = conf_threshold  # Confidence threshold
        self.nms_threshold = nms_threshold  # NMS threshold
        self.net_width = net_width  # The input image width of the network
        self.net_height = net_height  # The height image width of the network
        self.classes = self.get_names(classespath)
        self.yolo_model = self.get_yolo_model(modelpath, is_GPU)
        self.outputs_names = self.get_outputs_names()

    def get_names(self,classespath):
        # COCO Object category name
        classesFile = classespath
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        return classes

    def get_yolo_model(self, modelpath, is_GPU):
        cfg_file = modelpath[0]
        weights_file = modelpath[1]
        net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
        if not is_GPU:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            # Here is the NVIDIA GPU, and some other GPUs are also supported
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net

    def get_outputs_names(self):
        # All network layer names
        layersNames = self.yolo_model.getLayerNames()
        # Output network layer name, such as the network layer without connection output.
        return [layersNames[i[0] - 1] for i in
                self.yolo_model.getUnconnectedOutLayers()]

    def postprocess(self, img_cv2, outputs):
        # Post-processing
        # Use NMS to remove low-confidence bounding boxes
        img_height, img_width, _ = img_cv2.shape

        # Only keep the output bounding box of high confidence scores
        # Use the category label with the highest score as the category label of the bounding box
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * img_width)
                    center_y = int(detection[1] * img_height)
                    width = int(detection[2] * img_width)
                    height = int(detection[3] * img_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Use NMS to eliminate redundant overlapping bounding boxes of lower confidences
        indices = cv2.dnn.NMSBoxes(boxes,
                                   confidences,
                                   self.conf_threshold,
                                   self.nms_threshold)
        results = []
        for ind in indices:
            res_box = {}
            res_box["class_id"] = class_ids[ind[0]]
            res_box["score"] = confidences[ind[0]]

            box = boxes[ind[0]]
            res_box["box"] = (box[0],
                              box[1],
                              box[0] + box[2],
                              box[1] + box[3])
            results.append(res_box)

        return results

    def predict(self, img_cv2):
        img = img_cv2
        # Create a 4D blob of network input.
        blob = cv2.dnn.blobFromImage(img, 1 / 255,
                                     (self.net_width, self.net_height),
                                     swapRB=True,
                                     crop=False)
        # Set the input blob of the model
        self.yolo_model.setInput(blob)
        # Forward calculation
        outputs = self.yolo_model.forward(self.outputs_names)
        # Post-processing
        results = self.postprocess(img, outputs)

        return results

    def visualize(self, img, results):
        np.random.seed(200)
        COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        for result in results:
            color = [int(c) for c in COLORS[result["class_id"]]]
            left, top, right, bottom = result["box"]
            # Draw border
            cv2.rectangle(img,
                          (left, top),
                          (right, bottom),
                          color, 2)
            # The category name and confidence score of the bounding box
            label = '%.2f' % result["score"]
            class_id = result["class_id"]
            if self.classes:
                assert (result["class_id"] < len(self.classes))
                label = '%s:%s' % (self.classes[class_id], label)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            #Painting category and accuracy
            cv2.rectangle(img,
                          (left, top - round(1.5 * label_size[1])),
                          (left + round(1.5 * label_size[0]),top + baseline),
                          color,
                          cv2.FILLED)
            cv2.putText(img,
                        label,
                        (left, top),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 0),
                        1)
        # Calculate rate information
        # getPerfProfile() The function returns the total inference time of the model and
        # the time spent in each network layer (in layersTimes).
        # t, _ = self.yolov3_model.getPerfProfile()
        # label = 'Inference time: %.2f ms' % \
        #         (t * 1000.0 / cv2.getTickFrequency())
        # cv2.putText(img_cv2, label, (0, 15),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        return img