import time, shutil, os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# from datatools import input_data
import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)

# learning_rate = 0.01
# training_epochs = 1000
# batch_size = 100
# dislay_step = 1

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    return tf.nn.relu(tf.matmul(input, W) + b)

def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [784, 256], [256])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [256, 256], [256])
    with tf.variable_scope("output"):
        output = layer(hidden_2, [256, 10], [10])

    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)

    return loss

# def training(cost, global_step):
#     tf.summary.scalar("cost", cost)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     train_op = optimizer.minimize(cost, global_step=global_step)
#     return train_op

# def evaluate(output, y):
#     correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar("validation", accuracy)
#     return accuracy

with tf.Graph().as_default():
    # mnist data image of shape 28 * 28 = 784
    x = tf.placeholder(tf.float32, name="x", shape=[None, 784])
    y = tf.placeholder(tf.float32, name="y", shape=[None, 10])
    sess = tf.Session()

    # optimized starting point
    """
    Using a checkpoint file that we saved while training our original feed-forward network,
    we can re-instantiate the inference and loss components
    while also maintaining a list of pointers to the variables in the original graph
    for future use in var_list_opt (where opt stands for the optimal parameter settings):
    """
    print("optimized starting point");
    with tf.variable_scope("mlp_model") as scope:
        output_opt = inference(x)
        cost_opt = loss(output_opt, y)
        saver = tf.train.Saver()
        scope.reuse_variables()
        var_list_opt = [
            "hidden_1/W",
            "hidden_1/b",
            "hidden_2/W",
            "hidden_2/b",
            "output/W",
            "output/b" ]
        var_list_opt = [tf.get_variable(v) for v in var_list_opt]
        # saver.restore(sess, "frozen_mlp_checkpoint/model-checkpoint-547800")
        saver.restore(sess, r"C:\Users\Gyan Tatiya\Documents\GitHub\Fundamentals-of-Deep-Learning-Book-GT\chapter3\mlp_logs\model-checkpoint-162800")

    # random
    """
    Similarly, we can reuse the component constructors to create a randomly initialized network.
    Here we store the variables in var_list_rand for the next step of our program:
    """
    print("random starting point");
    with tf.variable_scope("mlp_init") as scope:
        output_rand = inference(x)
        cost_rand = loss(output_rand, y)
        scope.reuse_variables()
        var_list_rand = [
            "hidden_1/W",
            "hidden_1/b",
            "hidden_2/W",
            "hidden_2/b",
            "output/W",
            "output/b" ]
        var_list_rand = [tf.get_variable(v) for v in var_list_rand]
        init_op = tf.initialize_variables(var_list_rand)
        sess.run(init_op)

    # interpolate
    """
    With these two networks appropriately initialized, we can now construct the linear interpolation
    using the mixing parameters alpha and beta:
    """
    with tf.variable_scope("mlp_inter") as scope:
        alpha = tf.placeholder("float", [1, 1])
        beta = 1 - alpha

        # interpolate the weights and biases:w
        print("interpolate the weights and biases");
        h1_W_inter = var_list_opt[0] * beta + var_list_rand[0] * alpha
        h1_b_inter = var_list_opt[1] * beta + var_list_rand[1] * alpha
        h2_W_inter = var_list_opt[2] * beta + var_list_rand[2] * alpha
        h2_b_inter = var_list_opt[3] * beta + var_list_rand[3] * alpha
        o_W_inter = var_list_opt[4] * beta + var_list_rand[4] * alpha
        o_b_inter = var_list_opt[5] * beta + var_list_rand[5] * alpha

        # create the interpolated activations
        print("create the interpolated activations");
        h1_inter = tf.nn.relu(tf.matmul(x, h1_W_inter) + h1_b_inter)
        h2_inter = tf.nn.relu(tf.matmul(h1_inter, h2_W_inter) + h2_b_inter)
        o_inter = tf.nn.relu(tf.matmul(h2_inter, o_W_inter) + o_b_inter)

        # whats the error on the interpolated tensor
        cost_inter = loss(o_inter, y)

        # log the results
        print("Log the results");
        summary_writer = tf.summary.FileWriter("linear_interp_logs/", graph=sess.graph)
        summary_op = tf.summary.merge_all()

        print("Run interpolator")
        results = []

        """
        Finally, we can vary the value of alpha to understand how the error surface changes
        as we traverse the line between the randomly initialized point and the final SGD solution:
        """
        for a in np.arange(-2, 2, 0.01):
            print("Run interpolator: ", a)
            feed_dict = {
                x: mnist.test.images,
                y: mnist.test.labels,
                alpha: [[a]]
            }
            cost = sess.run(cost_inter, feed_dict=feed_dict)
            results.append(cost)
        
        print("plot the results")
        plt.plot(np.arange(-2, 2, 0.01), results, 'ro')
        plt.ylabel('Error Incurred')
        plt.xlabel('Alpha')
        plt.show()