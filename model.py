import uuid
import os
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


def train_model(x, y, embeddings, batch_size=128, epochs=10, lstm_size=64, model_path=os.path.join('model', 'model.ckpt')):
    graph = tf.Graph()
    run_id = uuid.uuid4().hex
    print('Creating graph ', run_id)

    sent_len = np.shape(x)[1]
    n_classes = np.shape(y)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))

        weights = tf.Variable(tf.truncated_normal((lstm_size, n_classes)))
        biases = tf.Variable(tf.truncated_normal([n_classes]))

        data = tf.nn.embedding_lookup(embeddings, inputs)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

        outputs, _ = tf.nn.dynamic_rnn(lstm, data, dtype=tf.float32)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        logits = tf.matmul(outputs, weights) + biases

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
            print('Model restored from ', model_path, '\nContinuing at epoch ', current_epoch.eval())

        print('Training...')
        for epoch in range(current_epoch.eval(), epochs):
            for data_index in range(len(y)):
                batch_x, batch_y = generate_batch(x, y, data_index, batch_size)

                _, loss_val, summary = sess.run((optimizer, loss, merged), {inputs: batch_x, labels: batch_y})
                writer.add_summary(summary)

                if data_index % 500 == 0:
                    print('Epoch: ', epoch, '    Step: ', data_index, '/', len(y), '\nLoss: ', loss_val)
            current_epoch.load(epoch + 1, sess)
            saver.save(sess, model_path)
            print('Model saved in ', model_path, ' after epoch ', epoch)


def test_model(x, y, embeddings, batch_size=128, lstm_size=64, model_path=os.path.join('model', 'model.ckpt')):
    assert os.path.exists('model')

    graph = tf.Graph()
    sent_len = np.shape(x)[1]
    n_classes = np.shape(y)[1]

    with graph.as_default():
        inputs = tf.placeholder(tf.int32, (None, sent_len))
        labels = tf.placeholder(tf.int32, (None, n_classes))

        weights = tf.Variable(tf.truncated_normal((lstm_size, n_classes)))
        biases = tf.Variable(tf.truncated_normal([n_classes]))

        data = tf.nn.embedding_lookup(embeddings, inputs)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

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
