#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/13 17:03
@Description: First, you should use the
tf_text_graph_ssd/faster_rcnn/mask_rcnn module under opencv/samples/dnn
to generate the pbtxt file. Only TensorFlow 1.X model files can be converted,
TensorFlow 2.X does not support such operations. But TensorFlow itself
provides a module to generate pbtxt. You can also implement target detection
through TenorFlow or PyTorch's advanced API.
"""
import cv2
import numpy as np

class detection_tfmodel(object):
    def __init__(self, modelpath, classespath, conf_threshold=0.5,nms_threshold=0.4,net_width=300, net_height=300, is_GPU=False):
        self.modelpath = modelpath
        self.classespath = classespath
        self.conf_threshold = conf_threshold  # Confidence threshold
        self.nms_threshold = nms_threshold  # NMS threshold
        self.net_width = net_width  # The input image width of the network
        self.net_height = net_height  # The height image width of the network
        self.classes = self.get_names()
        self.detection_model = self.get_detection_model(is_GPU)
        self.outputs_names = self.get_outputs_names()

    def get_names(self):
        # COCO Object category name
        classesFile = self.classespath
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        return classes

    def get_detection_model(self, is_GPU):
        pb_file = self.modelpath[0]
        pbtxt_file = self.modelpath[1]
        net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
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
        layersNames = self.detection_model.getLayerNames()
        # Output network layer name, such as the network layer without connection output.
        return [layersNames[i[0] - 1] for i in \
                self.detection_model.getUnconnectedOutLayers()]

    def postprocess(self, img_cv2, outputs):
        img_height, img_width, _ = img_cv2.shape
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output[0, 0]:
                # [batch_id, class_id, confidence, left, top, right, bottom]
                confidence = detection[2]
                if confidence > self.conf_threshold:
                    left = int(detection[3] * img_width)
                    top = int(detection[4] * img_height)
                    right = int(detection[5] * img_width)
                    bottom = int(detection[6] * img_height)
                    width = right - left + 1
                    height = bottom - top + 1

                    class_ids.append(int(detection[1]))
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
            res_box["box"] = (box[0], box[1], box[0] + box[2], box[1] + box[3])

            results.append(res_box)

        return results

    def predict(self, img_cv2):
        # Create a 4D blob of network input.
        blob = cv2.dnn.blobFromImage(
            img_cv2,
            size=(self.net_width, self.net_height),
            swapRB=True, crop=False)

        # Set the input blob of the model
        self.detection_model.setInput(blob)

        # Forward calculation
        outputs = self.detection_model.forward(self.outputs_names)

        # Post-processing
        results = self.postprocess(img_cv2, outputs)

        return results

    def visualize(self, img_cv2, results):
        np.random.seed(300)
        COLORS = np.random.randint(0, 255, size=(len(self.classes) , 3), dtype="uint8")
        for result in results:
            left, top, right, bottom = result["box"]
            color = [int(c) for c in COLORS[result["class_id"]]]
            cv2.rectangle(img_cv2,
                          (left, top),
                          (right, bottom),
                          color, 3)
            # Get the label for the class name and its confidence
            label = '%.2f' % result["score"]
            if self.classes:
                assert (result["class_id"] < len(self.classes))
                label = '%s:%s' % (self.classes[result["class_id"]], label)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            cv2.rectangle(
                img_cv2,
                (left, top - round(1.5 * label_size[1])),
                (left + round(1.5 * label_size[0]), top + baseline),
                color,
                cv2.FILLED)
            cv2.putText(img_cv2,
                        label,
                        (left, top),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 1)
        # Calculate rate information
        # getPerfProfile() The function returns the total inference time of the model and
        # the time spent in each network layer (in layersTimes).
        # t, _ = self.yolov3_model.getPerfProfile()
        # label = 'Inference time: %.2f ms' % \
        #         (t * 1000.0 / cv2.getTickFrequency())
        # cv2.putText(img_cv2, label, (0, 15),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        return img_cv2