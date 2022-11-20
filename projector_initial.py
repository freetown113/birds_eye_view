import cv2
import numpy as np
from typing import Callable, Tuple
import math


class Projector:
    '''This class contain necessary instruments to make an image
    projection i.e. undistort image, obtain transformation matrix,
    wart transformation.

    Takes object with the camera configuration as input
    config: configuration file that contains camera parameters
    '''
    def __init__(self,
                 config: Callable):
        self.config = config
        self.target_height = 250
        self.target_width = 250

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
        H = cv2.getPerspectiveTransform(*self.get_perspective_points())

        points = self.get_object_pixels(image)

        dst = cv2.perspectiveTransform(points, H)
        out = np.zeros((self.target_width, self.target_height, 1),
                       dtype=np.float32)

        output_plane = self.locate_points(dst, out)

        return output_plane

    def get_perspective_points(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_object_pixels(self,
                          image: np.ndarray
                          ) -> np.ndarray:
        '''This function get segmentation mask and return only coorinates
        of the points that describe the object of interest.
        '''
        image = np.max(image, axis=-1, keepdims=True)

        res = np.where(image.squeeze(axis=-1) == 255.)
        points = np.stack(res[::-1], axis=-1)

        points = np.array(points).reshape(-1, 1, 2).astype(np.float32)

        return points

    def locate_points(self,
                      projected_points: np.ndarray,
                      output_plane: np.ndarray
                      ) -> np.ndarray:
        '''This function locate projected points of perspective on the output
        plane in correct way.
        '''
        rotation = np.array([[math.cos(1.57), -math.sin(1.57)],
                             [math.sin(1.57), math.cos(1.57)]])

        flip = np.array([[1, 0],
                         [0, -1]])

        points = projected_points @ rotation @ flip

        dst = np.clip(np.ceil(points), 0, 249).astype(np.int32).reshape(-1, 2)
        output_plane[dst[:, 0], dst[:, 1], 0] = 255.

        return output_plane
