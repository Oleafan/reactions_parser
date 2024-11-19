'''Custom classes for images'''

#%% Imports

import cv2
import numpy as np

from PIL import Image


#%% Supporting functions

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    '''Transforms PIL Image to CV2 format'''
    
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    '''Transforms PIL Image to CV2 format'''
    
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


#%% Class and generators

class CV2Image(np.ndarray):
    '''Syntactic sugar for cv2 images'''
    
    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
    
    @property
    def image(self):
        return cv2_to_pil(self)
    
    def show(self):
        self.image.show()


def cv2image_from_file(path: str) -> CV2Image:
    '''Reads image file'''
    
    return CV2Image(cv2.imread(path))


def cv2image_from_pil(image: Image.Image) -> CV2Image:
    '''Trasforms PIL Image object into CV2 Image'''
    
    return CV2Image(pil_to_cv2(image))


