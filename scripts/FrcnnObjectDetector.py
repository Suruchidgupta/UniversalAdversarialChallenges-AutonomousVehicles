import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

class FrcnnObjectDetector(object):
    """
    This class handles loading and running Faster R-CNN model with class outputs from coco dataset
    The code is available at https://github.com/tensorflow/models/tree/master/research/object_detection/colab_tutorials
    """

    def __init__(self):
        """
        Constructor for the class to load the trained model
        """
        # patching few paths (tf1 into `utils.ops`, location of gfile)
        utils_ops.tf = tf.compat.v1
        tf.gfile = tf.io.gfile

        self.model = tf.saved_model.load('../models/frcnn_model/saved_model')
        self.PATH_TO_LABELS = '../models/models/research/object_detection/data/mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)
        # print(self.model.signatures['serving_default'].inputs)

    def model_run(self, image_path):
        """
        This method reads the image from the image_path, passes it for Object Detection and saves the output
        Args:
            image_path: file path for image to be processed

        Returns: None

        """
        # Reading the image and calling Object Detection
        image_np = np.array(Image.open(image_path))
        output_dict = self.process_frame(image_np)

        # Visualizing the results of detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=.7)    # Setting classification threshold as 0.7

        # objects = []
        # scores = output_dict['detection_scores']
        # for index, value in enumerate(output_dict['detection_classes']):
        #     object_dict = {}
        #     if scores[index] > 0.7:
        #         object_dict[(self.category_index.get(value)).get('name').encode('utf8')] = scores[index]
        #         objects.append(object_dict)
        # print(objects)

        # Saving the output in the results folder
        result = Image.fromarray(image_np)
        result_path = image_path.replace('data/', 'results/frcnn_model/')
        result.save(result_path)

    def process_frame(self, frame):
        """
        This method is taken from the colab tutorials for tensorflow object_detection repo on git
        It takes np array of image as input, converts it in tensor, process and returns the output tensor
        Args:
            frame: np array of the input image

        Returns: output tensor

        """
        image = np.asarray(frame)

        # Converting input into tensor and adding an axis as it expects a batch of images
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Running object detection
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # Ignoring new axis and taking first index for detection results
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bounding box mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        return output_dict