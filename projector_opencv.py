import cv2
import numpy as np
from typing import Callable, Tuple


class Projector:
    '''This class contain necessary instruments to make an image
    projection i.e. undistort image, obtain transformation matrix,
    wart transformation.

    Takes object with the camera configuration as input
    config: configuration file that contains camera parameters
    '''
    def __init__(self,
                 config: Callable) -> None:
        self.config = config

    def remove_optical_distortion(self,
                                  img: np.ndarray
                                  ) -> np.ndarray:
        """This function remove optical distortion. It takes image and
        camera parameters as input and returns undistorted image.

        img: input image
        """
        height, width = img.shape[:2]

        intrinsic = np.array([self.config.camera_matrix['data']],
                             dtype=np.float32).reshape(3, 3)
        distortion = np.array(self.config.distortion_coefficients['data'],
                              dtype=np.float32)
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion,
                                                        (width, height), 1,
                                                        (width, height))

        return cv2.undistort(img, newcameramtx, distortion)

    def make_projection(self,
                        image: np.ndarray,
                        ) -> np.ndarray:
        '''This function make projection of the provided image, based
        on perspective points from source image and corresponding points
        from traget image.

        image: input image to be transformed
        src_pts: designated 4 points from the source image
        dst_pts: corresponding 4 points from destinamion image
        '''
        H = cv2.getPerspectiveTransform(*self.get_points())
        bird_eye_view = cv2.warpPerspective(image, H, dsize=(250, 250))
        return bird_eye_view

    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        '''This function return 4 points that describe the object od interest
        on the original image and 4 poits that describe the rectangle where
        this object should be located after transformation.
        '''
        dst_pts = np.float32([[100., 20.],
                              [120., 20.],
                              [135., 230.],
                              [110., 230.]])
        src_pts = np.float32([[840., 725.],
                              [990., 725.],
                              [1720., 1208.],
                              [130., 1208.]])
        return (src_pts, dst_pts)
