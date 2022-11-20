import cv2
import numpy as np


class Processor:
    '''Auxilary class with a set of simple methods for image processing.
    '''
    def __init__(self,
                 ) -> None:
        pass

    def open(self,
             path_to_image: str
             ) -> np.ndarray:
        try:
            image = cv2.imread(path_to_image)
        except IOError as ioe:
            print(f'Open file failed: {ioe}')
        return image

    def show(self,
             image: np.ndarray,
             name: str = 'result'
             ) -> None:
        try:
            cv2.imshow(name, image)
        except FileNotFoundError as nfe:
            print(f'Exceptoin was raised while access file {nfe}')
        cv2.waitKey()

    def save(self,
             image: np.ndarray,
             path_to_save: str
             ) -> None:
        try:
            cv2.imwrite(path_to_save, image)
        except FileNotFoundError as nfe:
            print(f'Exceptoin was raised while access file {nfe}')
