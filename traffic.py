import cv2  # opencv module for image processing
import numpy as np  # numpy library
import os  # for getting the current working directory
import sys  # for command-line arguments
import tensorflow as tf  # tensorflow

from sklearn.model_selection import train_test_split  # train_test_split function


# Constants
EPOCHS = 10  # number of epochs to train
IMAGE_WIDTH = 30  # width of the image
IMAGE_HEIGHT = 30  # height of the image
NUM_CATEGORIES = 43  # number of categories
TEST_SIZE = 0.4  # test size


def main():  # main function
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:  # if there are no command-line arguments
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
        model.save(filename)  # save the model to file
        print(f"Model saved to {filename}.")  # print the model saved message


def load_data(data_dir):  # load data from directory
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMAGE_WIDTH x IMAGE_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []  # list of images
    labels = []  # list of labels

    # getting the root directory of all the data
    root = os.path.join(os.getcwd(), data_dir)

    # getting the list of all the subdirectories within the data folder each of which is a label for our model
    subdirs = sorted([int(i) for i in os.listdir(root)])  # sorted list of labels

    for subdir in subdirs:  # for each label
        # print(subdir) # print the label
        for filename in os.listdir(os.path.join(root, str(subdir))):
            # loading the image file using opencv module
            image = cv2.imread(os.path.join(root, str(subdir), filename))
            # resizing the image matrix to make suitable for feeding into neural net
            resized_image = cv2.resize(
                image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

            # storing the image array into a list
            images.append(resized_image)

            # as each subdir name indicates a category of sign so it's our label
            labels.append(subdir)

    return (images, labels)  # return the list of images and labels


def get_model():  # get a compiled neural network
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMAGE_WIDTH, IMAGE_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([  # create a sequential model
        # convolutional layer with 32 filters of size 3x3
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
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
    # categorical_crossentropy is used for multi-class classification
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model  # returning the model


if __name__ == "__main__":
    main()  # run the program
