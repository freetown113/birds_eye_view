from mmseg.apis import inference_segmentor, init_segmentor
import cv2
import numpy as np


class Segmentor:
    '''Class initalizes segmentation algorithm and pass parameters
    from config file.
    '''
    def __init__(self,
                 config) -> None:
        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(config.config_file, config.checkpoint, 
                                    device=config.device)

    def segment_image(self, image):
        '''Handle given image by segmentation algorithm. The segmentation
        model inference segmentation mask of one precise class - Road.  
        '''
        result = inference_segmentor(self.model, image)

        classes = np.unique(result[0])

        for i in classes:
            if i == 91:  # return only pixels belong to class 'Road'
                output = result[0] == i
                output = output.astype(np.uint8)*255

        return np.expand_dims(output, axis=-1)
