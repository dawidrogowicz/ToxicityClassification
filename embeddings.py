from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.recurrent import lstm
from tflearn import regression
from tflearn.models import DNN
import tensorflow as tf
import numpy as np
import collections
import random
import math
import matplotlib.pyplot as plt

data_index = 0

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


def model(shape):
    net = input_data(shape)
    net = lstm(net, 128)
    net = fully_connected(net, 6)
    net = regression(net)

    return DNN(net, tensorboard_dir='log')


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)

        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]

        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def train_embeddings(data, vocabulary_size, reverse_dictionary):
    graph = tf.Graph()

    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, [batch_size])
        train_labels = tf.placeholder(tf.int32, (batch_size, 1))
        valid_dataset = tf.constant(valid_examples, tf.int32)

        # with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform((vocabulary_size, embedding_size), -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal((vocabulary_size, embedding_size),
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    num_steps = 100001

    with tf.Session(graph=graph) as sess:
        init.run()
        print('TensorFlow initialized')

        average_loss = 0

        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = sess.run((optimizer, loss), feed_dict)
            average_loss += loss_val

            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 2000

                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # if step % 2000 == 0:
            #     sim = similarity.eval()
            #     for i in range(valid_size):
            #         valid_word = reverse_dictionary.get(valid_examples[i], 'UNKNOWN')
            #         top_k = 8
            #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            #         log_str = 'Nearest to %s:' % valid_word
            #         for k in range(top_k):
            #             close_word = reverse_dictionary.get(nearest[k], 'UNKNOWN')
            #             log_str = '%s %s,' % (log_str, close_word)
            #         print(log_str)
        return normalized_embeddings.eval()


def plot_embeddings(embeddings, labels, filename):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
    plt.show()
