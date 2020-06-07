#!python

import tesseract_east
import keras_craft


def transcribe_text(path_to_image, model):
    """Takes in an image and returns the text detected in it"""
    print(model.text_recognition(path_to_image))


if __name__ == '__main__':
    transcribe_text('./images/example_03.jpg', tesseract_east)
    transcribe_text('./images/example_03.jpg', keras_craft)
