from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.recurrent import lstm
from tflearn import regression
from tflearn.models import DNN


def model(shape):
    net = input_data(shape)
    net = lstm(net, 128)
    net = fully_connected(net, 6)
    net = regression(net)

    return DNN(net, tensorboard_dir='log')
