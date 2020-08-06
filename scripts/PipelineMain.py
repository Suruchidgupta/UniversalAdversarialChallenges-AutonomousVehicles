from scripts.DirectoryReader import DirectoryReader
from scripts.YoloObjectDetector import YoloObjectDetector
from scripts.FrcnnObjectDetector import FrcnnObjectDetector

if __name__ == "__main__":
    DR = DirectoryReader()
    filepath_dict = DR.get_all_filepaths()

    # Processing all the images through YOLO model
    YOD = YoloObjectDetector()
    for folder, file_path in filepath_dict.items():
        print('Processing folder - YOLO '+folder)
        images = DR.read_images(file_path)
        YOD.model_run(images)

    # Processing all the images through Faster R-CNN model
    FOD = FrcnnObjectDetector()
    for folder, file_path in filepath_dict.items():
        print('Processing folder - FRCNN '+folder)
        for path in file_path:
            try:
                FOD.model_run(path)
            except:
                print('Execution failed for ', path)
