import tensorflow as tf

SENTENCE_LSTM = 10
WORD_DIM = 1000
CLASS_DIM = 2
RNN_SIZE = 120
LEARNING_RATE = 0.01


class Model:
    def __init__(self):
        # 1
        self.input_data = tf.placeholder(tf.float32, [None, SENTENCE_LSTM, WORD_DIM])
        self.output_data = tf.placeholder(tf.float32, [None, SENTENCE_LSTM, CLASS_DIM])
        weight, bias = weight_and_bias(2 * RNN_SIZE, CLASS_DIM)

        # 2
        fw_cell = tf.nn.rnn_cell.LSTMCell(RNN_SIZE, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.LSTMCell(RNN_SIZE, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        # 3
        output, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unpack(tf.transpose(self.input_data, perm=[1, 0, 2])))
        # 4
        output = tf.reshape(tf.transpose(tf.pack(output), perm=[1, 0, 2]), [-1, 2 * RNN_SIZE])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)

        # 5
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.output_data))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

        # 6
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.output_data, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def weight_and_bias(n_hidden, n_classes):
    weight = tf.random_normal([2 * n_hidden, n_classes])
    bias = tf.random_normal([n_classes])
    return tf.Variable(weight), tf.Variable(bias)
