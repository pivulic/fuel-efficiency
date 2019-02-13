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

# Tail the data
auto_set = AutoSet()

# Clean the data, since column Horsepower contains 6 errors (?)
auto_set.drop_missing_rows()

# Convert Origin column to numeric columns for each country
auto_set.convert_origin_to_numeric_columns()

# Split the data into train and test
auto_set.split_data()
