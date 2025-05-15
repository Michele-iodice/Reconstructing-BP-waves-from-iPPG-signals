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

    def is_frame_blurry(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < 50

    def extract_holistic(self, videoFileName, scale_percent=50, frame_interval=10):
        """
        This method computes the RGB-mean signal using the whole skin (holistic).

        Args:
            videoFileName (str): video file name or path.
            scale_percent (int): Percentage to scale down the video resolution for faster processing.

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
        max_frames_limit = 10000
        selected_indices = list(range(17, 27)) + [1, 2, 3, 14, 15, 16]

        for frame in extract_frames_yield(videoFileName, frame_interval=frame_interval):
            print(processed_frames_count)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            processed_frames_count += 1

            landmarks = fa.get_landmarks(resized_image)

            if landmarks is not None:
                landmarks = landmarks[0]
                ldmks = np.zeros((68, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0

                for idx in selected_indices:
                    if idx >= len(landmarks):
                        continue
                    x_pixel = int(landmarks[idx][0] * 100 / scale_percent)
                    y_pixel = int(landmarks[idx][1] * 100 / scale_percent)
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

            if processed_frames_count >= max_frames_limit:
                print("[WARNING] Limit,max number of frame reached.")
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
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        return image
