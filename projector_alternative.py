import cv2
import math
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
                 config: Callable):
        self.config = config
        self.camera_height = 1.68
        self.target_height = 250
        self.target_width = 250
        self.scale = 0.2

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

    def get_camera_params(self,
                          config: Callable
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the intrinsic and extrinsic parameters
        Returns:
            Camera extrinsic and intrinsic matrices
        """
        x = -50
        y = -22
        z = self.camera_height

        intrinsic = np.array(self.config.camera_matrix['data'],
                             dtype=np.float32).reshape(3, 3)
        intrinsic = np.row_stack([np.column_stack([intrinsic, [0, 0, 0]]),
                                 [0, 0, 0, 1]])

        translation = np.identity(4)
        translation[:3, 3] = (-x, -y, -z)

        # Rotate to camera coordinates
        rotation = np.array([[0., -1., 0., 0.],
                            [0., 0., -1., 0.],
                            [1., 0., 0., 0.],
                            [0., 0., 0., 1.]])
        RT = rotation @ translation

        return RT, intrinsic

    def make_projection(self, image):

        RT, intrinsic = self.get_camera_params(self.config)
        P = intrinsic @ RT

        # Flip y points positive upwards
        img_plane = Image(self.target_height, self.target_width, self.scale)
        plane = img_plane.get_plane()
        plane[1] = -plane[1]

        pixel_coords = self.perspective(plane, P, self.target_height,
                                        self.target_width)
        bird_eye_view = self.bilinear_sampler(image, pixel_coords)

        return bird_eye_view.astype(np.uint8)

    def bilinear_sampler(self, imgs, pix_coords):
        """
        Construct a new image by bilinear sampling from the input image.
        Args:
            imgs:                   [H, W, C]
            pix_coords:             [h, w, 2]
        :return:
            sampled image           [h, w, c]
        """
        img_h, img_w, img_c = imgs.shape
        pix_h, pix_w, pix_c = pix_coords.shape
        out_shape = (pix_h, pix_w, img_c)

        pix_x, pix_y = np.split(pix_coords, [1], axis=-1)  # [pix_h, pix_w, 1]
        pix_x = pix_x.astype(np.float32)
        pix_y = pix_y.astype(np.float32)

        # Rounding
        pix_x0 = np.floor(pix_x)
        pix_x1 = pix_x0 + 1
        pix_y0 = np.floor(pix_y)
        pix_y1 = pix_y0 + 1

        # Clip within image boundary
        y_max = (img_h - 1)
        x_max = (img_w - 1)
        zero = np.zeros([1])

        pix_x0 = np.clip(pix_x0, zero, x_max)
        pix_y0 = np.clip(pix_y0, zero, y_max)
        pix_x1 = np.clip(pix_x1, zero, x_max)
        pix_y1 = np.clip(pix_y1, zero, y_max)

        # Weights [pix_h, pix_w, 1]
        wt_x0 = pix_x1 - pix_x
        wt_x1 = pix_x - pix_x0
        wt_y0 = pix_y1 - pix_y
        wt_y1 = pix_y - pix_y0

        # indices in the image to sample from
        dim = img_w

        # Apply the lower and upper bound pix coord
        base_y0 = pix_y0 * dim
        base_y1 = pix_y1 * dim

        # 4 corner vertices
        idx00 = (pix_x0 + base_y0).flatten().astype(np.int)
        idx01 = (pix_x0 + base_y1).astype(np.int)
        idx10 = (pix_x1 + base_y0).astype(np.int)
        idx11 = (pix_x1 + base_y1).astype(np.int)

        # Gather pixels from image using vertices
        imgs_flat = imgs.reshape([-1, img_c]).astype(np.float32)
        im00 = imgs_flat[idx00].reshape(out_shape)
        im01 = imgs_flat[idx01].reshape(out_shape)
        im10 = imgs_flat[idx10].reshape(out_shape)
        im11 = imgs_flat[idx11].reshape(out_shape)

        # Apply weights [pix_h, pix_w, 1]
        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1
        output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
        return output

    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self,
                                    theta: Tuple[float, float, float]
                                    ) -> np.ndarray:

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def perspective(self, cam_coords, proj_mat, h, w):
        """
        This method make transformation of
        Args:
            cam_coords:         [4, npoints]
            proj_mat:           [4, 4]

        Returns:
            pix coords:         [h, w, 2]
        """
        rotation = np.transpose(self.eulerAnglesToRotationMatrix(
                                (0., 0., self.camera_height)))
        eps = 1e-7
        rotation = np.row_stack([np.column_stack([rotation, [0, 0, 0]]),
                                 [0, 0, 0, 1]])
        proj_mat = proj_mat @ rotation
        pix_coords = proj_mat @ cam_coords

        pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + eps)
        pix_coords = np.reshape(pix_coords, (2, h, w))
        pix_coords = np.transpose(pix_coords, (1, 2, 0))
        return pix_coords


class Image:
    """
    Defines a plane in the world
    """

    def __init__(self, height, width, scale):
        self.h, self.w = height, width
        self.scale = scale

    def get_plane(self):
        """
        Returns:
            Grid coordinate: [b, 3/4, row*cols]
        """
        xmin = 0
        xmax = xmin + self.h * self.scale
        ymin = 0
        ymax = ymin + self.w * self.scale

        x = np.linspace(xmin, xmax, self.h)
        y = np.linspace(ymin, ymax, self.w)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)

        coords = np.stack([x, y, z, np.ones_like(x)], axis=0)
        return coords
