import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import collections
import random
import math
import os
import uuid

data_index = 0
batch_size = 256
embedding_size = 128
skip_window = 4
num_skips = 8
num_sampled = 32


def generate_batch(data):
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


def train_embeddings(data, vocabulary_size, n_steps=400000):
    graph = tf.Graph()
    run_id = uuid.uuid4().hex
    print('Creating graph ', run_id)

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

        tf.summary.scalar('embeddings loss', loss)

        global_step = tf.Variable(0, False)
        lr = tf.train.exponential_decay(1.0, global_step, 100000, .96, staircase=True)

        tf.summary.scalar('embeddings learning rate', lr)

        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = tf.div(embeddings, norm)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join('log', 'embeddings', run_id), graph)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        print('TensorFlow initialized')

        average_loss = 0

        for step in range(n_steps):
            batch_inputs, batch_labels = generate_batch(data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val, summary = sess.run((optimizer, loss, merged), feed_dict)
            average_loss += loss_val

            writer.add_summary(summary, step)

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 2000

                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        return normalized_embeddings.eval()


def visualize_embeddings(lexicon, embed_lookup):
    embeds = []
    labels = []
    for i, label in enumerate(lexicon):
        labels.append(label)
        embeds.append(embed_lookup[i])
        if i > 5000:
            break

    if not os.path.exists(os.path.join('log', 'projector')):
        os.makedirs(os.path.join('log', 'projector'))

    embeddings = tf.Variable(np.array(embeds), name='embeddings')
    meta_path = os.path.join('log', 'projector', 'metadata.tsv')
    embeddings_path = os.path.join('log', 'projector', 'embeddings.ckpt')

    with open(meta_path, 'w') as f:
        for label in labels:
            f.write('%s\n' % label)

    with tf.Session() as sess:
        saver = tf.train.Saver([embeddings])
        sess.run(embeddings.initializer)
        saver.save(sess, embeddings_path)

        writer = tf.summary.FileWriter(os.path.join('log', 'projector'))
        config = projector.ProjectorConfig()

        embed = config.embeddings.add()
        embed.tensor_name = embeddings.name
        embed.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(writer, config)

    print('embeddings visualised in tensorboard')
