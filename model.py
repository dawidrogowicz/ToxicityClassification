import uuid
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def generate_batch(x, y, data_index, batch_size=128):
    assert len(x) == len(y)

    first_index = data_index
    data_index = first_index + batch_size
    last_index = len(y) if data_index > len(y) else data_index

    batch = x[first_index:last_index]
    labels = y[first_index:last_index]

    return batch, labels


def get_cell(cell_size):
    cell = tf.nn.rnn_cell.GRUCell(cell_size)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=.8)



def train_model(x, y, embeddings, batch_size=128, epochs=10, cell_size=128,
                model_path=os.path.join('model', 'model.ckpt')):
    graph = tf.Graph()
    run_id = uuid.uuid4().hex
    print('Creating graph ', run_id)

    sent_len = np.shape(x)[1]
    n_classes = np.shape(y)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))

        # weights = tf.Variable(tf.truncated_normal((cell_size * 2, n_classes)))
        # biases = tf.Variable(tf.truncated_normal([n_classes]))

        data = tf.nn.embedding_lookup(embeddings, inputs)

        cell = get_cell(cell_size)

        rnn, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, data, dtype=tf.float32)
        rnn = tf.concat(rnn, -1)
        # rnn = tf.transpose(rnn, (1, 0, 2))
        # rnn = tf.gather(rnn, int(rnn.get_shape()[0]) - 1)
        # rnn = tf.matmul(rnn, weights) + biases

        avg_pool = tf.keras.layers.GlobalAvgPool1D()(rnn)
        max_pool = tf.keras.layers.GlobalMaxPool1D()(rnn)
        pool = tf.concat([avg_pool, max_pool], -1)

        logits = tf.contrib.layers.fully_connected(pool, n_classes)

        correct = tf.equal(tf.cast(tf.round(logits), dtype=tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        current_epoch = tf.Variable(0)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join('log', 'model', run_id), graph)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        saver = tf.train.Saver()

        if os.path.exists('model'):
            saver.restore(sess, model_path)
            print('Model restored from ', model_path,
                  '\nContinuing at epoch ', current_epoch.eval())

        print('Training...')
        for epoch in range(current_epoch.eval(), epochs):
            t_start = time.time()
            for data_index in range(len(y)):
                batch_x, batch_y = generate_batch(x, y, data_index, batch_size)

                _, loss_val, summary = sess.run((optimizer, loss, merged),
                                                {inputs: batch_x, labels: batch_y})
                writer.add_summary(summary)

                if data_index % 500 == 0:
                    current_time = time.time() - t_start
                    avg_loop_time = current_time / (data_index + 1)
                    loops_left = len(y) - data_index - 1
                    time_left = int(avg_loop_time * loops_left)
                    time_left = '{}h:{}m:{}s'.format(time_left // 3600,
                                                     time_left % 3600 // 60, time_left % 3600 % 60)
                    print('Epoch: {}    Step: {}/{}\nLoss: {}    Time left: {}'
                          .format(epoch, data_index, len(y), loss_val, time_left))

            current_epoch.load(epoch + 1, sess)
            saver.save(sess, model_path)
            print('Model saved in ', model_path, ' after epoch ', epoch)


def test_model(x, y, embeddings, batch_size=128, cell_size=64,
               model_path=os.path.join('model', 'model.ckpt')):
    assert os.path.exists('model')

    graph = tf.Graph()
    sent_len = np.shape(x)[1]
    n_classes = np.shape(y)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))

        weights = tf.Variable(tf.truncated_normal((cell_size, n_classes)))
        biases = tf.Variable(tf.truncated_normal([n_classes]))

        data = tf.nn.embedding_lookup(embeddings, inputs)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(cell_size)

        outputs, _ = tf.nn.dynamic_rnn(lstm, data, dtype=tf.float32)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        logits = tf.matmul(outputs, weights) + biases

        correct = tf.equal(tf.cast(tf.round(logits), dtype=tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        saver = tf.train.Saver()

        saver.restore(sess, model_path)
        print('Testing model restored from ', model_path)

        accuracies = []
        for data_index in tqdm(range(len(y))):
            batch_x, batch_y = generate_batch(x, y, data_index, batch_size)

            accuracy_val = sess.run(accuracy, {inputs: batch_x, labels: batch_y})
            accuracies.append(accuracy_val)

        avg_acc = np.mean(accuracies)
        print('Average accuracy: ', avg_acc)
