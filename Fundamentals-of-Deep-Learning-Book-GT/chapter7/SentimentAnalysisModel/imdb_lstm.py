import tensorflow as tf
#from lstm import LSTMCell
import read_imdb_data as data

training_epochs = 1000
batch_size = 32
display_step = 1

"""
First, we’ll want to map each word in the input review to a word vector.
To do this, we’ll utilize an embedding layer, which is a simple lookup table
that stores an embedding vector that corresponds to each word.
"""
def embedding_layer(input, weight_shape):
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    E = tf.get_variable("E", weight_shape, initializer=weight_init)
    # E_exp = tf.expand_dims(E, 0)
    # E_tiled= tf.tile(E_exp, [32, 1, 1])
    # return tf.batch_matmul(input, E_exp)
    incoming = tf.cast(input, tf.int32)
    embeddings = tf.nn.embedding_lookup(E, incoming)
    return embeddings

"""
We then take the result of the embedding layer and build an LSTM with dropout.
We do some extra work to pull out the last output emitted by the LSTM using the
tf.slice and tf.squeeze operators, which find the exact slice that contains the last output
of the LSTM and then eliminates the unnecessary dimension.
The change in dimensions is as follows:
[batch_size, max_time, cell.output_size] to [batch_size, 1, cell.output_size] to [batch_size, cell.output_size].
"""
def lstm(input, hidden_dim, keep_prob, phase_train):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * 2, state_is_tuple=True)
        lstm_outputs, state = tf.nn.dynamic_rnn(dropout_lstm, input, dtype=tf.float32)
        #return tf.squeeze(tf.slice(lstm_outputs, [0, tf.shape(lstm_outputs)[1]-1, 0], [tf.shape(lstm_outputs)[0], 1, tf.shape(lstm_outputs)[2]]))
        return tf.reduce_max(lstm_outputs, reduction_indices=[1])

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))

"""
We top it all off using a batch-normalized hidden layer, identical to the ones we’ve used time
and time again in previous examples.
Stringing all of these components together, we can build the inference graph:
"""
def inference(input, phase_train):
    embedding = embedding_layer(input, [30000, 512])
    lstm_output = lstm(embedding, 512, 0.5, phase_train)
    output = layer(lstm_output, [512, 2], [2], phase_train)
    return output

def loss(output, y):
    #xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    #train_loss_summary_op = tf.scalar_summary("train_cost", loss)
    train_loss_summary_op = tf.summary.scalar("train_cost", loss)
    #val_loss_summary_op = tf.scalar_summary("val_cost", loss)
    val_loss_summary_op = tf.summary.scalar("val_cost", loss)
    return loss, train_loss_summary_op, val_loss_summary_op

def training(cost, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
        use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary_op = tf.summary.scalar("accuracy", accuracy)
    return accuracy, accuracy_summary_op

if __name__ == '__main__':

    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            x = tf.placeholder("float", [None, 500])
            y = tf.placeholder("float", [None, 2])
            phase_train = tf.placeholder(tf.bool)

            output = inference(x, phase_train)

            cost, train_loss_summary_op, val_loss_summary_op = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op, eval_summary_op = evaluate(output, y)

            saver = tf.train.Saver(max_to_keep=100)

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

            #summary_writer = tf.train.SummaryWriter("imdb_lstm_logs/", graph=sess.graph)
            summary_writer = tf.summary.FileWriter("imdb_lstm_logs/", graph_def=sess.graph_def)

            #init_op = tf.initialize_all_variables()
            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            for epoch in range(training_epochs):

                avg_cost = 0.
                total_batch = int(data.train.num_examples/batch_size)
                print("Total of %d minbatches in epoch %d" % (total_batch, epoch))
                # Loop over all batches
                for i in range(total_batch):
                    minibatch_x, minibatch_y = data.train.minibatch(batch_size)
                    # Fit training using batch data
                    _, new_cost, train_summary = sess.run([train_op, cost, train_loss_summary_op], feed_dict={x: minibatch_x, y: minibatch_y, phase_train: True})
                    summary_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost/total_batch
                    print("Training cost for batch %d in epoch %d was:" % (i, epoch), new_cost)
                    if i % 100 == 0:
                        print("Epoch:", '%04d' % (epoch+1), "Minibatch:", '%04d' % (i+1), "cost =", "{:.9f}".format((avg_cost * total_batch)/(i+1)))
                        val_x, val_y = data.val.minibatch(data.val.num_examples)
                        val_accuracy, val_summary, val_loss_summary = sess.run([eval_op, eval_summary_op, val_loss_summary_op], feed_dict={x: val_x, y: val_y, phase_train: False})
                        summary_writer.add_summary(val_summary, sess.run(global_step))
                        summary_writer.add_summary(val_loss_summary, sess.run(global_step))
                        print("Validation Accuracy:", val_accuracy)

                        saver.save(sess, "imdb_lstm_logs/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)
                # Display logs per epoch step
                # if epoch % display_step == 0:
                #     print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)
                #     val_x, val_y = data.val.minibatch(data.val.num_examples)
                #     val_accuracy, val_summary, val_loss_summary = sess.run([eval_op, eval_summary_op, val_loss_summary_op], feed_dict={x: val_x, y: val_y, phase_train: False})
                #     summary_writer.add_summary(val_summary, sess.run(global_step))
                #     summary_writer.add_summary(val_loss_summary, sess.run(global_step))
                #     print "Validation Accuracy:", val_accuracy
                #
                #     saver.save(sess, "imdb_lstm_logs/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)


            print("Optimization Finished!")
