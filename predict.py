from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('TkAgg')
import pathlib

import pandas
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers


class AutoSet:
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

# Tail the data
auto_set = AutoSet()

# Clean the data, since column Horsepower contains 6 errors (?)
auto_set.drop_missing_rows()

# Convert Origin column to numeric columns for each country
auto_set.convert_origin_to_numeric_columns()
print(auto_set.dataset.tail())
