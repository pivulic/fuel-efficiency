from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
import pandas

from tensorflow import keras
from tensorflow.keras import layers


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


class AutoSet:
    train_dataset = None
    test_dataset = None
    train_labels = None
    test_labels = None
    normed_train_dataset = None
    normed_test_dataset = None

    LABEL = 'MPG'
    EPOCHS = 1000

    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        self.columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
        self.dataset = pandas.read_csv(self.url, names=self.columns,  na_values="?", comment='\t', sep=" ", skipinitialspace=True)

    def drop_missing_rows(self):
        self.dataset = self.dataset.dropna()

    def convert_origin_to_numeric_columns(self):
        origin = self.dataset.pop('Origin')
        self.dataset['USA'] = (origin == 1) * 1.0
        self.dataset['Europe'] = (origin == 2) * 1.0
        self.dataset['Japan'] = (origin == 3) * 1.0

    def split_data(self):
        self.train_dataset = self.dataset.sample(frac=0.8, random_state=0)
        self.test_dataset = self.dataset.drop(self.train_dataset.index)

    def separate_label(self):
        self.train_labels = self.train_dataset.pop(self.LABEL)
        self.test_labels = self.test_dataset.pop(self.LABEL)

    def get_training_statistics(self):
        train_statistics = self.train_dataset.describe()
        return train_statistics.transpose()

    def normalize(self):
        train_statistics = self.get_training_statistics()
        self.normed_train_dataset = (self.train_dataset - train_statistics['mean']) / train_statistics['std']
        self.normed_test_dataset = (self.test_dataset - train_statistics['mean']) / train_statistics['std']

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation=tensorflow.nn.relu, input_shape=[len(self.train_dataset.keys())]),
            layers.Dense(64, activation=tensorflow.nn.relu),
            layers.Dense(1)
        ])

        optimizer = tensorflow.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def train_model(self):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        return model.fit(
            self.normed_train_dataset, self.train_labels,
            epochs=self.EPOCHS, validation_split=0.2, verbose=0,
            callbacks=[early_stop, PrintDot()]
        )

    def plot_history(self, history):
        hist = pandas.DataFrame(history.history)
        hist['epoch'] = history.epoch

        pyplot.figure()
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Mean Abs Error [MPG]')
        pyplot.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
        pyplot.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
        pyplot.legend()
        pyplot.ylim([0, 5])

        pyplot.figure()
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Mean Square Error [$MPG^2$]')
        pyplot.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        pyplot.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
        pyplot.legend()
        pyplot.ylim([0, 20])
        pyplot.show()

    def plot_error_distribution(self, model):
        test_predictions = model.predict(self.normed_test_dataset).flatten()
        error = test_predictions - self.test_labels
        pyplot.hist(error, bins=25)
        pyplot.xlabel("Prediction Error [MPG]")
        pyplot.ylabel("Count")
        pyplot.show()


# Tail the data
auto_set = AutoSet()

# Clean the data, since column Horsepower contains 6 errors (?)
auto_set.drop_missing_rows()

# Convert Origin column to numeric columns for each country
auto_set.convert_origin_to_numeric_columns()

# Split the data into train and test
auto_set.split_data()

# Split features from labels
auto_set.separate_label()

# Normalize the data
auto_set.normalize()

# Build the model
model = auto_set.build_model()

# Train the model
history = auto_set.train_model()
auto_set.plot_history(history)

# Measure model accuracy
loss, mean_abs_error, mean_squared_error = model.evaluate(auto_set.normed_test_dataset, auto_set.test_labels, verbose=0)
print("Testing accuracy (Mean Abs Error): {:5.2f} MPG".format(mean_abs_error))

# Plot prediction error distribution
auto_set.plot_error_distribution(model)
