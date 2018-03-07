from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.recurrent import lstm
from tflearn import regression
from tflearn.models import DNN
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import collections
import random
import math
import os

data_index = 0
batch_size = 256
embedding_size = 200
skip_window = 4
num_skips = 8
num_sampled = 32


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


def train_embeddings(data, vocabulary_size):
    graph = tf.Graph()

    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, [batch_size])
        train_labels = tf.placeholder(tf.int32, (batch_size, 1))

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

        init = tf.global_variables_initializer()

    num_steps = 300001

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

        return normalized_embeddings.eval()


def visualize_embeddings(embeds, labels):
    embeddings = tf.Variable(embeds, name='embeddings')
    meta_path = os.path.join('log', 'metadata.tsv')
    embeddings_path = os.path.join('log', 'embeddings.ckpt')

    with open(meta_path, 'w') as f:
        for label in labels:
            f.write('%s\n' % label)

    with tf.Session() as sess:
        saver = tf.train.Saver([embeddings])
        sess.run(embeddings.initializer)
        saver.save(sess, embeddings_path)

        writer = tf.summary.FileWriter('log')
        config = projector.ProjectorConfig()

        embed = config.embeddings.add()
        embed.tensor_name = embeddings.name
        embed.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(writer, config)
