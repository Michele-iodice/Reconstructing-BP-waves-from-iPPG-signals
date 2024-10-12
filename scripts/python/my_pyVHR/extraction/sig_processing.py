from my_pyVHR.extraction.utils import *
from my_pyVHR.extraction.skin_extraction_methods import *
from my_pyVHR.extraction.sig_extraction_methods import *
import cv2
import numpy as np
import face_alignment
import torch

"""
This module defines classes or methods used for Signal extraction and processing.
"""


class SignalProcessing:
    """
        This class performs offline signal extraction with different methods:

        - holistic.

        - squared / rectangular patches.
    """

    def __init__(self):
        # Common parameters #
        self.tot_frames = None
        self.visualize_skin_collection = []
        self.skin_extractor = SkinExtractionConvexHull('CPU')
        # Patches parameters #
        high_prio_ldmk_id, mid_prio_ldmk_id = get_magic_landmarks()
        self.ldmks = high_prio_ldmk_id + mid_prio_ldmk_id
        self.square = None
        self.rects = None
        self.visualize_skin = False
        self.visualize_landmarks = False
        self.visualize_landmarks_number = False
        self.visualize_patch = False
        self.font_size = 0.3
        self.font_color = (255, 0, 0, 255)
        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []

    def set_total_frames(self, n):
        """
        Set the total frames to be processed; if you want to process all the possible frames use n = 0.
        
        Args:  
            n (int): number of frames to be processed.
            
        """
        if n < 0:
            print("[ERROR] n must be a positive number!")
        self.tot_frames = int(n)

    def set_skin_extractor(self, extractor):
        """
        Set the skin extractor that will be used for skin extraction.
        
        Args:  
            extractor: instance of a skin_extraction class (see :py:mod:`pyVHR.extraction.skin_extraction_methods`).
            
        """
        self.skin_extractor = extractor

    def extract_holistic(self, videoFileName):
        """
        This method computes the RGB-mean signal using the whole skin (holistic).

        Args:
            videoFileName (str): video file name or path.

        Returns:
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels].
                             The second dimension is 1 because the whole skin is considered as one estimator.
        """
        self.visualize_skin_collection = []
        skin_ex = self.skin_extractor

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

        sig = []
        processed_frames_count = 0

        for frame in extract_frames_yield(videoFileName):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames_count += 1
            width = image.shape[1]
            height = image.shape[0]

            landmarks = fa.get_landmarks(image)

            if landmarks is not None:
                landmarks = landmarks[0]
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0

                for idx in range(min(len(landmarks), 468)):
                    x_pixel = int(landmarks[idx][0])
                    y_pixel = int(landmarks[idx][1])
                    ldmks[idx, 0] = y_pixel
                    ldmks[idx, 1] = x_pixel

                cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, ldmks)
            else:
                cropped_skin_im = np.zeros_like(image)
                full_skin_im = np.zeros_like(image)

            if self.visualize_skin == True:
                self.visualize_skin_collection.append(full_skin_im)

            sig.append(holistic_mean(
                cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH),
                np.int32(SignalProcessingParams.RGB_HIGH_TH)))

            if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                break

        sig = np.array(sig, dtype=np.float32)
        return sig

    def draw_landmarks(image, landmarks):
        """
        Function to draw facial landmark on image
        Args:
            image (ndarray): image.
            landmarks (ndarray): landmark coordinate (x, y).
        """
        for (x, y) in landmarks:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)  # Colore verde per i punti
        return image
