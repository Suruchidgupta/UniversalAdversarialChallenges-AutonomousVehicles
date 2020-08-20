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
        dir_names = [x[0] for x in os.walk("../data")]
        filepath_dict = defaultdict()

        # Adding the filepath in each folder as an array
        for x in dir_names[1:]:
            # Ignoring the directories 'Advanced Scenarios' and 'Semantic Segmentation'
            if not (x.__contains__('Advanced Scenarios') or x.__contains__('Semantic Segmentation')):
                filepaths = glob.glob(x + "/*.*")
                filepath_dict[x] = filepaths

        return filepath_dict
