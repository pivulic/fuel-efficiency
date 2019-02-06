# fuel-efficiency

Deep learning with TensorFlow to predict fuel efficiency: https://www.tensorflow.org/tutorials/keras/basic_regression
## Pre-requisites

1. Install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html):

    ```
    pip3 install virtualenvwrapper
    ```

1. Create virtual environment in `~/.virtualenvs`:

    ```
    mkvirtualenv fuel-efficiency
    ```

1. Install packages:

    ```
    pip install --requirement requirements.txt
    ```

## Run

1. Activate virtual environment:

    ```
    workon fuel-efficiency
    ```

1. Run:

    ```
    python predict.py
    ```
