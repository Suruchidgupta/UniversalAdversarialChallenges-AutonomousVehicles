import os
import glob
import cv2 as cv
from collections import defaultdict


class DirectoryReader(object):
    """
    This class handles the file/directory reading operation and returns an list of images or filepaths as required
    """

    @staticmethod
    def read_images(file_paths):
        """
        This method reads the list of filepaths and returns the image files packed with filepath
        Args:
            file_paths: list of filepaths

        Returns: list of filepaths and corresponding images

        """
        images = list()
        # Read all filepaths and add to list of images
        for name in file_paths:
            image = cv.imread(name)
            images.append((name, image))

        return images

    def get_all_filepaths(self):
        """
        This method reads all the directories in the /data folder and
        returns the filepaths of all the files in each of the directory
        Returns: dict with folder and all filepaths in each of the folders

        """
        # Reading all the folders in data directory
        dir_names = [x[0] for x in os.walk("../data")]  # TODO ['', '../data/New']
        filepath_dict = defaultdict()

        # Adding the filepath in each folder as an array
        for x in dir_names[1:]:
            filepaths = glob.glob(x + "/*.*")
            filepath_dict[x] = filepaths

        # Testing code
        # filepaths = glob.glob('../data/Vehicle Art and Textures/OWMF6.jpg')
        # x = '../data/Vehicle Art and Textures/'
        # filepath_dict[x] = filepaths
        # print(filepaths)

        return filepath_dict
