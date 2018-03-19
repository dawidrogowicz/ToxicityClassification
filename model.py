import uuid
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

cell_size = 512
batch_size = 64
model_path = os.path.join('model', 'model.ckpt')

def generate_batch(x, y, step):
    assert len(x) == len(y)

    first_index = step * batch_size
    data_index = first_index + batch_size
    last_index = len(y) if data_index > len(y) else data_index

    batch = x[first_index:last_index]
    labels = y[first_index:last_index]

    return batch, labels


def get_cell():
    cell = tf.nn.rnn_cell.GRUCell(cell_size)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=.8)

def train_model(x, y, embeddings, n_epochs=10):
    graph = tf.Graph()
    run_id = uuid.uuid4().hex
    print('Creating graph ', run_id)

    sent_len = np.shape(x)[1]
    n_classes = np.shape(y)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))
        labels = tf.cast(labels, tf.float32)

        data = tf.nn.embedding_lookup(embeddings, inputs)

        rnn, _ = tf.nn.bidirectional_dynamic_rnn(get_cell(), get_cell(), data, dtype=tf.float32)
        rnn = tf.concat(rnn, -1)

        max_pool = tf.layers.max_pooling1d(rnn, sent_len, sent_len)
        max_pool = tf.squeeze(max_pool, [1])

        fully_connected = tf.contrib.layers.fully_connected(max_pool, 128)
        dropout = tf.layers.dropout(fully_connected, .9)

        logits = tf.contrib.layers.fully_connected(dropout, n_classes,
                                                   activation_fn=None)
        correct = tf.equal(tf.round(tf.sigmoid(logits)), labels)
        accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(correct, -1), dtype=tf.float32))

        tf.summary.scalar('accuracy', accuracy)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        current_epoch = tf.Variable(0)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join('log', 'model', run_id), graph)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        saver = tf.train.Saver()

        if os.path.exists('model') and os.listdir('model'):
            saver.restore(sess, model_path)
            print('Model restored from ', model_path,
                  '\nContinuing at epoch ', current_epoch.eval())

        n_steps = len(y) // batch_size + 1

        print('Training ', run_id, '...')
        for epoch in range(current_epoch.eval(), n_epochs):
            t_start = time.time()

            for step in range(n_steps):
                batch_x, batch_y = generate_batch(x, y, step)
                feed_dict = {inputs: batch_x, labels: batch_y}

                _, loss_val = sess.run((optimizer, loss), feed_dict)

                if step % 64 == 0:
                    acc, summary = sess.run((accuracy, merged), feed_dict)
                    writer.add_summary(summary, epoch * n_steps + step)

                    current_time = time.time() - t_start
                    avg_loop_time = current_time / (step + 1)
                    loops_left = n_steps - step - 1
                    n_epochs_left = n_epochs - epoch
                    time_left = int(avg_loop_time * loops_left * n_epochs_left)
                    time_left = '{}h:{}m:{}s'.format(time_left // 3600,
                                                     time_left % 3600 // 60, time_left % 3600 % 60)
                    print('Epoch: {}    Step: {}/{}    Time left: {}\nLoss: {}    Accuracy: {}'
                          .format(epoch, step, n_steps, time_left, loss_val, acc))

            current_epoch.load(epoch + 1, sess)
            saver.save(sess, model_path)
            print('Model saved in ', model_path, ' after epoch ', epoch)


def test_model(x, y, embeddings):
    assert os.path.exists('model') and os.listdir('model')

    graph = tf.Graph()
    sent_len = np.shape(x)[1]
    n_classes = np.shape(y)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))
        labels = tf.cast(labels, tf.float32)

        data = tf.nn.embedding_lookup(embeddings, inputs)

        rnn, _ = tf.nn.bidirectional_dynamic_rnn(get_cell(), get_cell(), data, dtype=tf.float32)
        rnn = tf.concat(rnn, -1)

        max_pool = tf.layers.max_pooling1d(rnn, sent_len, sent_len)
        max_pool = tf.squeeze(max_pool, [1])

        fully_connected = tf.contrib.layers.fully_connected(max_pool, 128)
        dropout = tf.layers.dropout(fully_connected, .9)

        logits = tf.contrib.layers.fully_connected(dropout, n_classes,
                                                   activation_fn=tf.sigmoid)
        correct = tf.equal(tf.round(logits), labels)
        accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(correct, -1), dtype=tf.float32))

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        saver = tf.train.Saver()

        saver.restore(sess, model_path)
        print('Testing model restored from ', model_path)

        false_logits = []
        accuracies = []
        for step in range(len(y) // batch_size + 1):
            batch_x, batch_y = generate_batch(x, y, step)
            feed_dict = {inputs: batch_x, labels: batch_y}

            logits_val, corr = sess.run((logits, correct), feed_dict)
            if step % 16 == 0:
                for i in range(15):
                    for j, logit_correct in enumerate(corr[i]):
                        if not logit_correct:
                            false_logits.append(batch_y[i][j])
                            print(['{:.1f}'.format(x) for x in logits_val[i]], '    ',
                                  batch_y[i], ' => ', corr[i])

            accuracy_val = sess.run(accuracy, feed_dict)
            accuracies.append(accuracy_val)

        avg_acc = np.mean(accuracies)
        print('Average accuracy: ', avg_acc, 'FP/FN: ', np.mean(false_logits))


def predict(x, embeddings, n_classes=6):
    assert os.path.exists('model') and os.listdir('model')

    graph = tf.Graph()
    sent_len = np.shape(x)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))
        labels = tf.cast(labels, tf.float32)

        data = tf.nn.embedding_lookup(embeddings, inputs)

        rnn, _ = tf.nn.bidirectional_dynamic_rnn(get_cell(), get_cell(), data, dtype=tf.float32)
        rnn = tf.concat(rnn, -1)

        max_pool = tf.layers.max_pooling1d(rnn, sent_len, sent_len)
        max_pool = tf.squeeze(max_pool, [1])

        fully_connected = tf.contrib.layers.fully_connected(max_pool, 128)
        dropout = tf.layers.dropout(fully_connected, .9)

        logits = tf.contrib.layers.fully_connected(dropout, n_classes,
                                                   activation_fn=tf.sigmoid)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        saver = tf.train.Saver()

        saver.restore(sess, model_path)
        print('Making predictions from model at ', model_path)

        prediction = []
        for step in range(len(x) // batch_size + 1):
            batch_x, batch_y = generate_batch(x, x, step)
            feed_dict = {inputs: batch_x}
            logits_val = sess.run(logits, feed_dict)
            prediction.extend(logits_val)

        return prediction
