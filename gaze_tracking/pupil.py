import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 15, 75, 75)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        
        # --- NEW LOGIC: Find the most circular contour ---
        if not contours:
            return

        best_contour = None
        min_circularity_error = float('inf')

        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area < 5: continue # Ignore noise
            if perimeter == 0: continue
            
            # Circularity = 4*pi*Area / Perimeter^2. Perfect circle = 1.0
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            # We want circularity close to 1.0
            error = abs(1.0 - circularity)
            
            if error < min_circularity_error:
                min_circularity_error = error
                best_contour = c
        
        if best_contour is None:
            # Fallback to old method if no circle found
            contours = sorted(contours, key=cv2.contourArea)
            best_contour = contours[-2] if len(contours) > 1 else contours[0]

        try:
            moments = cv2.moments(best_contour)
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass
