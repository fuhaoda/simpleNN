import pickle
import gzip
import numpy as np


def encode_label(j):  # <1>
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    # <1> We one-hot encode indices to vectors of length 10.


def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]  # <1>
    labels = [encode_label(y) for y in data[1]]  # <2>
    return zip(features, labels)  # <3>


def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:  # <4>
        train_data, validation_data, test_data = pickle.load(f, encoding='bytes')  # <5>

    return shape_data(train_data), shape_data(test_data)

# <1> We flatten the input images to feature vectors of length 784. The matrix will be a 784 by sample size n matrix.
# <2> All labels are one-hot encoded.
# <3> Then we create pairs of features and labels. zip returns an iterator which can only be consumed once.
# <4> Unzipping and loading the MNIST data yields three data sets.
# <5> We discard validation data here and reshape the other two data sets.
