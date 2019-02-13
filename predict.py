from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('TkAgg')
import pathlib

import pandas
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers


class AutoSet:
    train_dataset = None
    test_dataset = None
    normed_train_dataset = None
    normed_test_dataset = None

    label = 'MPG'

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
        self.train_dataset.pop(self.label)
        self.test_dataset.pop(self.label)

    def get_training_statistics(self):
        train_statistics = self.train_dataset.describe()
        return train_statistics.transpose()

    def normalize(self):
        train_statistics = self.get_training_statistics()
        self.normed_train_dataset = (self.train_dataset - train_statistics['mean']) / train_statistics['std']
        self.normed_test_dataset = (self.test_dataset - train_statistics['mean']) / train_statistics['std']

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
print(auto_set.normed_train_dataset.tail())
print(auto_set.normed_test_dataset.tail())
