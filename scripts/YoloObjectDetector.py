import cv2
import numpy as np


class YoloObjectDetector(object):
    """
    This class handles loading and running YOLO v3 model with class outputs from coco dataset
    """

    def __init__(self):
        """
        Constructor for the class loads the weights for the models and reads the output classnames
        """

        # Loading the pre-trained model
        self.model = cv2.dnn.readNet('../models/yolo_model/yolov3.weights', '../models/yolo_model/yolov3.cfg')
        self.layer_names = self.model.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        self.classes = self.readClassNames('../models/yolo_model/coco.names')

    def readClassNames(self, classFile):
        """
        This method read the output classnames from the given filepath
        :param classFile: path to the file
        :return: list of classes
        """
        with open(classFile, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def model_run(self, image_path, results_array):
        """
        This method call the ObjectDetection for each of the image
        :param image_path: filepath for each image
        :param results_array: dataframe to store the individual results
        :return:
        """
        image = cv2.imread(image_path)
        self.process_frame(image_path, image, results_array)

    def process_frame(self, name, frame, results_array):
        """
        The method processes each frame(image) and saves the images with bounding boxes in the /results folder
        :param name: filepath required to save the output
        :param frame: image file to be processed
        :param results_array: dataframe to store the individual results
        :return: None
        """

        h, w, channels = frame.shape
        color = 1
        font = cv2.FONT_HERSHEY_COMPLEX

        # Resizing image for YOLO
        resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
        blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Setting up input, forward pass through YOLO
        self.model.setInput(blob)
        outputs = self.model.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []

        # Looping through outputs for objects detected
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Considering the classification threshold as 0.7
                if confidence > 0.7:
                    # Calculating coordinates for bounding box
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Adding the bounding boxes for objects
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                label = label + ": " + str(round(confidences[i] * 100, 2))

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label, (x + 2, y - 5), font, 0.5, color, 2)

                # Collating the results into a dataframe
                results_array.loc[name, self.classes[class_ids[i]]] = results_array.loc[
                                                                          name, self.classes[class_ids[i]]] + 1

        # Saving the output image in the results folder
        path = name.replace('data/', 'results/yolo_model/')
        filename = f'{path}'
        cv2.imwrite(filename, frame)
