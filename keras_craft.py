import matplotlib.pyplot as plt
import keras_ocr


# keras-ocr automatically downloads the pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()


def text_recognition(path_to_image):
    # the pipline takes in a list of images, image_paths, or urls
    images = [keras_ocr.tools.read(path_to_image)]
    #images = [keras_ocr.tools.read(url) for url in images]

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)
    return [group[0] for group in prediction_groups[0]]

    # Plot the predictions
    # fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    # for ax, image, predictions in zip(axs, images, prediction_groups):
    #     keras_ocr.tools.drawAnnotations(
    #         image=image, predictions=predictions, ax=ax)


if __name__ == '__main__':
    print("Testing Keras OCR Model")
    print('_______Results_______')
    recognize_text('./images/example_04.jpg')
