import pandas as pd
from collections import defaultdict
from scripts.DirectoryReader import DirectoryReader
from scripts.YoloObjectDetector import YoloObjectDetector
from scripts.FrcnnObjectDetector import FrcnnObjectDetector
from scripts.EvaluatePerformance import EvaluatePerformance

if __name__ == "__main__":
    # Adding the results dict and coco class names to collate the results
    results_dict = {'YOLO': {}, 'FRCNN': {}}
    coco_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet',
                    'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush']

    # Reading the ground truth data as pandas dataframe
    excel_df = pd.ExcelFile('../GroundTruth.xlsx')

    # Creating a dict for all the sheets, filling Nan with 0 and setting index as Image name
    ground_truth = {'../data/' + sheet_name: excel_df.parse(sheet_name).fillna(0)
                    for sheet_name in excel_df.sheet_names}
    for name, values in ground_truth.items():
        values.set_index('Image', inplace=True)

    # Instantiating DirectoryReader and reading all the files from /data directory
    DR = DirectoryReader()

    filepath_dict = DR.get_all_filepaths()

    # Creating object for EvaluatePerformance and dict to store all scores
    eval_perf = EvaluatePerformance()
    scores = defaultdict()

    # Processing all the images through YOLO model
    YOD = YoloObjectDetector()
    for folder, file_path in filepath_dict.items():
        print('Processing folder - YOLO ' + folder)

        # Creating dataframe to store results for Object Detection
        results_dict['YOLO'][folder] = pd.DataFrame(index=file_path, columns=coco_classes)
        results_dict['YOLO'][folder] = results_dict['YOLO'][folder].fillna(0)

        # Running Object Detection for each file_path
        for path in file_path:
            try:
                YOD.model_run(path, results_dict['YOLO'][folder])
            except:
                print('Execution failed - YOLO : ', path)

        # Computing the evaluation metrics for the predictions
        results_dict['YOLO'][folder] = eval_perf.compute_scores(ground_truth[folder], results_dict['YOLO'][folder])

    # Printing evaluation metrics for YOLO
    print('YOLO model Results: ')
    eval_perf.publish_results(results_dict['YOLO'])

    # Processing all the images through Faster R-CNN model
    FOD = FrcnnObjectDetector()
    for folder, file_path in filepath_dict.items():
        print('Processing folder - FRCNN ' + folder)

        # Creating dataframe to store results for Object Detection
        results_dict['FRCNN'][folder] = pd.DataFrame(index=file_path, columns=coco_classes)
        results_dict['FRCNN'][folder] = results_dict['FRCNN'][folder].fillna(0)

        # Running Object Detection for each file_path
        for path in file_path:
            try:
                FOD.model_run(path, results_dict['FRCNN'][folder])
            except:
                print('Execution failed - FRCNN : ', path)

        # Computing the evaluation metrics for the predictions
        results_dict['FRCNN'][folder] = eval_perf.compute_scores(ground_truth[folder], results_dict['FRCNN'][folder])

    # Printing evaluation metrics for FRCNN
    print('FRCNN model Results: ')
    eval_perf.publish_results(results_dict['FRCNN'])
