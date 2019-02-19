import argparse
import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
import numpy as np
from typing import Iterable, Tuple
import os

_CSV_FILENAME = 'driving_log.csv'

def read_train_data_in_batches(data_dir: str, csv_filename: str, batch_size=None) -> Iterable[Tuple[np.array, np.array]]:
    """Reads the train data from the given `data_dir` in batches.

    Args:
        data_dir: directory where the data is located at.
        csv_filename: name of the CSV training file inside `data_dir`.
        batch_size: is specified, we return data in batches.

    Returns:
        Generator of tuples, where each tuple contains the X data and the
        y data.
    """
    assert(batch_size is None or batch_size > 0)

    filepath = os.path.join(data_dir, csv_filename)
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)
        # Sanitize the header just in case
        assert(header[:4] == ['center', 'left', 'right', 'steering'])

        images = []
        measurements = []
        for line in reader:
            filenames = (os.path.join(data_dir, x.strip()) for x in line[:3]) 
            center, left, right = filenames
            measurement = float(line[3])

            # TODO: let's add left and right images with the right correction
            # for the steering measurement.
            images.append(cv2.imread(center))
            measurements.append(measurement)

            if batch_size and len(images) >= batch_size:
                yield images, measurements

                images.clear()
                measurements.clear()

        # It's possible that we still have some leftover data that hasn't been
        # reported, or maybe the batch_size is None.
        if images:
            yield images, measurements

def read_train_data(data_dir: str, csv_filename: str) -> Tuple[np.array, np.array]:
    """Reads the train data from the given `data_dir`.

    Args:
        data_dir: directory where the data is located at.
        csv_filename: name of the CSV training file inside `data_dir`.

    Returns:
        Tuples, where each tuple contains the X data and the y data.
    """
    filepath = os.path.join(data_dir, csv_filename)
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)
        # Sanitize the header just in case
        assert(header[:4] == ['center', 'left', 'right', 'steering'])

        images = []
        measurements = []
        for line in reader:
            filenames = (os.path.join(data_dir, x.strip()) for x in line[:3]) 
            center, left, right = filenames
            measurement = float(line[3])

            # TODO: let's add left and right images with the right correction
            # for the steering measurement.
            images.append(cv2.imread(center))
            measurements.append(measurement)

        return np.array(images), np.array(measurements)

def main(args):
    X_train, y_train = read_train_data(args.data_dir, _CSV_FILENAME)

    model = Sequential()
    # The following standardization worked better than the (X - mean) / stddev
    # standardization technique.
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    # We use mean squared error instead of something like softmax because we
    # are trying to predict a continuous value.
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=args.epochs)

    if args.model_filename:
        model.save(args.model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Trains a behavioral cloning model')
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int)
    parser.add_argument('-d', '--data_dir', required=True,
            help='Directory where the training data is present')
    parser.add_argument('-m', '--model_filename',
            help='If specified, the model is saved into the given location')
    parser.add_argument('-e', '--epochs', default=10, type=int,
            help='Numbers of epochs we should train our model with')

    args = parser.parse_args()
    main(args)
