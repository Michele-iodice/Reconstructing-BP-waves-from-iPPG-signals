import numpy as np
import os
from my_pyVHR.datasets.dataset import Dataset
from BP.BP import BPsignal


class BP4D(Dataset):
    """
        BP4D+ Dataset

        .. BP4D+ dataset structure:
        .. -----------------
        ..     datasetBP4D+/
        ..     |
        ..     |F001/
        ..       |
        ..       |T1/
        ..        |
        ..        |-- videoSample.avi
        ..        |-- signalGT.xml
        """
    name = 'BP4D'
    # path relative where the dataset was located on your filesystem
    signalGT = 'BP'  # GT signal type
    numLevels = 3  # depth of the filesystem collecting video and BP files
    numSubjects = 27  # number of subjects
    video_EXT = 'avi'  # extension of the video files
    frameRate = 250  # vieo frame rate
    VIDEO_SUBSTRING = 'vid'  # substring contained in the filename
    SIG_EXT = 'txt'  # extension of the BP files
    SIG_SUBSTRING = 'BP_mmHg'  # substring contained in the filename
    SIG_SampleRate = 1000  # sample rate of the BP files

    def loadFilenames(self):
        """Load dataset file names: define vars videoFilenames and BPFilenames."""

        # -- loop on the dir struct of the dataset getting filenames
        for root, dirs,  files in os.walk(self.videodataDIR):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)

                # -- select video
                if filename.endswith(self.video_EXT) and (name.find(self.VIDEO_SUBSTRING) >= 0):
                    self.videoFilenames.append(filename)

        # -- loop on the dir struct of the dataset getting BP filenames
        for root, dirs, files in os.walk(self.videodataDIR):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)
                # -- select signal
                if filename.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING) >= 0):
                    self.sigFilenames.append(filename)

        # -- number of videos
        self.numVideos = len(self.videoFilenames)

    def getVideoFilename(self, video_idx=0):
        """Get video file name given the progressive index."""
        return self.videoFilenames[video_idx]

    def getSigFilename(self, video_idx=0):
        """Get Signal file name given the progressive index."""
        return self.sigFilenames[video_idx]

    def getVideoFilenames(self):
        """Get an array contain the videos file name"""
        return self.videoFilenames

    def getSigFilenames(self):
        """Get an array contain the signals file name"""
        return self.sigFilenames

    def len_dataset(self):
        """Get the size of the dataset"""
        return len(self.videoFilenames)

    def readSigfile(self, filename):
        """
        Load signal from file.

        Returns:
            a :class:`BP_Estimator.BPsignal` object that can be used to extract BPM signal from ground truth BP signal.
        """
        bp_trace= []
        with open(filename, 'r') as f:
            bp_data = [float(line.strip()) for line in f.readlines()]

        bp_trace= np.array(bp_data)

        return BPsignal(bp_trace, self.SIG_SampleRate)
