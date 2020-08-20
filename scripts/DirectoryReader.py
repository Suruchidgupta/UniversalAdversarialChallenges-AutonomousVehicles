import os
import glob
from collections import defaultdict


class DirectoryReader(object):
    """
    This class handles the file/directory reading operation and returns an list of images or filepaths as required
    """

    def get_all_filepaths(self):
        """
        This method reads all the directories in the /data folder and
        returns the filepaths of all the files in each of the directory
        Returns: dict with folder and all filepaths in each of the folders

        """
        # Reading all the folders in data directory
        # TODO [x[0] for x in os.walk("../data")]
        dir_names = ['', '../data/Art-in-surrounding and Murals']
        filepath_dict = defaultdict()

        # Adding the filepath in each folder as an array
        for x in dir_names[1:]:
            if not (x.__contains__('Advanced Scenarios') or x.__contains__('Semantic Segmentation')):
                filepaths = glob.glob(x + "/*.*")
                filepath_dict[x] = filepaths

        # Testing code
        # filepaths = glob.glob('../data/Street Signs/6I6BJ.jpg')
        # x = '../data/Street Signs/'
        # filepath_dict[x] = filepaths
        # print(filepaths)

        return filepath_dict
