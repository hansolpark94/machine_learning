import tensorflow as tf

SENTENCE_LSTM = 10
WORD_DIM = 1000
CLASS_DIM = 2
RNN_SIZE = 120
LEARNING_RATE = 0.01

def train():
    # 1
    train_in, train_out = get_train_data()
    test_in, test_out = get_test_data()
    DL = Model()
    batch_size = 50

    # 2
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for e in range(10):
            for ptr in range(0, len(train_in), batch_size):
                # 3
                sess.run(DL.optimizer, {DL.input_data: train_in[ptr:ptr + batch_size],
                                        DL.output_data: train_out[ptr:ptr + batch_size]})
            # 4
            acc = sess.run(DL.accuracy, feed_dict={DL.input_data: test_in, DL.output_data: test_out})
            loss = sess.run(DL.cost, feed_dict={DL.input_data: test_in, DL.output_data: test_out})

            print("Iter " + e + ", Minibatch Loss= " + str(loss) + ", Training Accuracy= " + str(acc))
