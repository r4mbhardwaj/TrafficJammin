import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # getting the root directory of all the data
    root = os.path.join(os.getcwd(), data_dir)
    # getting the list of all the subdirectories within the data folder each of which is a label for our model
    subdirs = sorted([int(i) for i in os.listdir(root)])

    for subdir in subdirs:
        print(subdir)
        for filename in os.listdir(os.path.join(root, str(subdir))):
            # loading the image file using opencv module
            img = cv2.imread(os.path.join(root, str(subdir), filename))
            # resizing the image matrix to make suitable for feeding into neural net
            resized_img = cv2.resize(
                img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            # storing the image array into a list
            images.append(resized_img)

            # as each subdir name indicates a category of sign so it's our label
            labels.append(subdir)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # convolutional layer with 32 filters of size 3x3
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # max pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D(2, 2),
        # convolutional layer with 64 filters of size 3x3
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # max pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D(2, 2),
        # convolutional layer with 128 filters of size 3x3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # max pooling layer with 2x2 pool size
        tf.keras.layers.MaxPooling2D(2, 2),
        # flattening the layer to make it suitable for fully connected layer
        tf.keras.layers.Flatten(),
        # fully connected layer with 512 units
        tf.keras.layers.Dense(512, activation="relu"),
        # fully connected layer with NUM_CATEGORIES units
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        # softmax activation function
    ])
    # compiling the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model


if __name__ == "__main__":
    main()
