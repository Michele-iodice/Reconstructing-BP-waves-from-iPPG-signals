import mediapipe as mp
from my_pyVHR.extraction.utils import *
from my_pyVHR.extraction.skin_extraction_methods import *
from my_pyVHR.extraction.sig_extraction_methods import *

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
        This method compute the RGB-mean signal using the whole skin (holistic);

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels]. The second dimension is 1 because
            the whole skin is considered as one estimators.
        """
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                # face landmarks
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    # skin extraction
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                # sig computing
                sig.append(holistic_mean(
                    cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                # loop break
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig
