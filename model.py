import argparse
import csv
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Iterable, List, Tuple
import os

# TODO: Maybe pass this as a flag?
_CSV_FILENAME = 'driving_log.csv'

def read_log_lines(data_dir: str, csv_filename: str) -> List[str]:
    """Reads the train data entries from the specified csv_filename."""
    filepath = os.path.join(data_dir, csv_filename)
    # TODO: maybe use pandas in order to read the data from csv.
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)
        # Sanitize the header just in case
        assert(header[:4] == ['center', 'left', 'right', 'steering'])

        lines = []
        for line in reader:
            lines.append(line)

        return lines


def read_data(
        data_dir: str,
        lines: List[str],
        augment: bool=False,
        correction: float=0.2,
        batch_size: int=600) -> Iterable[Tuple[np.array, np.array]]:
    """Reads the data from the given `data_dir` using the given log entries.

    Args:
        data_dir: directory where the data is located at.
        lines: lines read from the CSV log file.
        augment: whether we should generate more synthetic data or not.
        correction: correction factor for left and right images.
        batch_size: upper bound on how many elements we should return per
            batch.

    Yields:
        Iterator of list of tuples of size <= batch_size, where each tuple
        contains the X data and the y data.
    """

    num_samples = len(lines)

    imgs = []
    ms = []
    identity_fn = lambda img, m: (img, m)
    transformations = [identity_fn]
    if augment:
        # The only transformation function that we have right now is to mirror
        # the image horizontally.
        flip_fn = lambda img, m: (np.fliplr(img), -m)
        transformations.append(flip_fn)

    # This generator should never stop producing results.
    while True:
        for sample_line in lines:
            # For each entry we have three images: center, left and right, each
            # with a different offset value.
            filenames = (
                    os.path.join(data_dir, x.strip()) for x in sample_line[:3])
            offsets = (0.0, correction, -correction)

            for filename, offset in zip(filenames, offsets):
                img = cv2.imread(filename)
                measurement = float(sample_line[3]) + offset

                for transformation in transformations:
                    new_img, new_measurement = transformation(img, measurement)
                    imgs.append(new_img)
                    ms.append(new_measurement)

                    if len(imgs) == batch_size:
                        yield shuffle(np.array(imgs), np.array(ms))

                        imgs = []
                        ms = []


def main(args):
    lines = read_log_lines(args.data_dir, _CSV_FILENAME)
    # TODO: maybe add a flag for the test size.
    train_lines, validation_lines = train_test_split(lines, test_size=0.2)
    train_generator = read_data(
            args.data_dir,
            train_lines,
            augment=args.augment,
            correction=args.correction,
            batch_size=args.batch_size)
    validation_generator = read_data(
            args.data_dir,
            validation_lines,
            augment=args.augment,
            correction=args.correction,
            batch_size=args.batch_size)

    model = Sequential()
    # The following standardization worked better than the (X - mean) / stddev
    # standardization technique.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Nvidia CNN
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(args.dropout_rate))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    if args.summary:
        print(model.summary())

    # We use mean squared error instead of something like softmax because we
    # are trying to predict a continuous value.
    model.compile(loss='mse', optimizer='adam')
    # TODO: This may fall out of sync if we add more transformations.
    # Maybe the best idea is to generate a list of structs instead of lines,
    # where each class contains the log entry and the expected transformation.
    FACTOR = 6 if args.augment else 3
    steps_per_epoch = int(ceil(FACTOR * len(train_lines) / args.batch_size))
    validation_steps = int(ceil(FACTOR * len(validation_lines) / args.batch_size))
    history_object = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)

    if args.graph_loss:
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    if args.model_filename:
        model.save(args.model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Trains a behavioral cloning model')
    parser.add_argument('-a', '--augment', action='store_true',
            help='Whether to augment the data set or not')
    parser.add_argument('-b', '--batch_size', default=600, type=int,
            help='Batch size')
    parser.add_argument('-c', '--correction', default=0.2, type=float,
            help='Correction factor')
    parser.add_argument('-d', '--data_dir', required=True,
            help='Directory where the training data is present')
    parser.add_argument('-e', '--epochs', default=10, type=int,
            help='Numbers of epochs we should train our model with')
    parser.add_argument('-l', '--graph_loss', action='store_true',
            help='Whether we should graph the loss function or not')
    parser.add_argument('-m', '--model_filename',
            help='If specified, the model is saved into the given location')
    parser.add_argument('-r', '--dropout_rate', default=0.5, type=float,
            help='Dropout rate for dropout layers')
    parser.add_argument('-s', '--summary', action='store_true',
            help='Whether to print a summary of the model or not')

    args = parser.parse_args()
    main(args)
