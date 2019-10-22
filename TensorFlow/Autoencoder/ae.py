import tensorflow as tf


def classifier(my_classifier, x_train_temp, x_test_temp, y_train_temp, y_test_temp):
    """
    Train a classifier on test data and return accuracy and prediction on test data
    :param my_classifier:
    :param x_train_temp:
    :param x_test_temp:
    :param y_train_temp:
    :param y_test_temp:
    :return: accuracy, prediction
    """
    # Fit the model on the training data.
    my_classifier.fit(x_train_temp, y_train_temp)

    # See how the model performs on the test data.
    accuracy = my_classifier.score(x_test_temp, y_test_temp)
    prediction = my_classifier.predict(x_test_temp)
    probability = my_classifier.predict_proba(x_test_temp)

    return accuracy, prediction, probability


class EncoderDecoderNetwork:
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
            training_epochs=1000,
    ):
        """
        Implement an encoder decoder network and train it

        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.training_epochs = training_epochs

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')

        self.code_prediction = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))

        # Define optimizer
        with tf.name_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes ) +1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_" +str(i))
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_" +str(i))
            net = tf.layers.dense(inputs=net, units=self.n_dims_code)
        return net

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="decoder_" +str(i))
            #net = tf.layers.dense(inputs=net, units=self.output_channels, name="decoder_final")
            net = tf.layers.dense(inputs=net, units=self.output_channels, activation=tf.sigmoid, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, logs_path):
        """
        Train using provided data

        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path

        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(self.training_epochs):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict={self.X: x_data, self.Y: y_data})

            cost_log.append(c)

            # Print generated data after every 100 epoch
            # if (epoch + 1) % 100 == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
            #     generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})
            #     print("Generated: ")
            #     print(list(generated_output[0]))
            #     print("Original: ")
            #     print(list(y_data[0]))

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate_code(self, x_data):
        """
        Generate target robot data using source robot data

        :param x_data: source robot data

        :return: generated target robot data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_code = self.sess.run(self.code_prediction, feed_dict={self.X: x_data})

        return generated_code

    def generate(self, x_data):
        """
        Reconstruct input by passing through encoder and decoder

        :param x_data: input data

        :return: generated input data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss

class EncoderDecoderNetwork_BN:
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
            training_epochs=1000,
    ):
        """
        Implement an encoder decoder network and train it

        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.training_epochs = training_epochs

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')
        self.training_phase = tf.placeholder(tf.bool, None, name='training_phase')

        self.code_prediction = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))

        # Define optimizer
        # control dependency to ensure that batchnorm parameters are also updated
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        	self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes ) +1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i-1], activation=None, name="encoder_" +str(i))
                    net = tf.layers.batch_normalization(net, training=self.training_phase)
                    net = self.activation_fn(net)
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=None, name="encoder_" +str(i))
                    net = tf.layers.batch_normalization(net, training=self.training_phase)
                    net = self.activation_fn(net)
            net = tf.layers.dense(inputs=net, units=self.n_dims_code)
        return net

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=None, name="decoder_" +str(i))
                net = tf.layers.batch_normalization(net, training=self.training_phase)
                net = self.activation_fn(net)
            net = tf.layers.dense(inputs=net, units=self.output_channels, activation=None, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, logs_path):
        """
        Train using provided data

        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path

        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(self.training_epochs):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict={self.X: x_data, self.Y: y_data, self.training_phase: True})

            cost_log.append(c)

            # Print generated data after every 100 epoch
            # if (epoch + 1) % 100 == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
            #     generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})
            #     print("Generated: ")
            #     print(list(generated_output[0]))
            #     print("Original: ")
            #     print(list(y_data[0]))

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate_code(self, x_data, large_batch):
        """
        Generate latent code using the input

        :param x_data: input data
        :param small_batch: boolean value to indicate small batch size

        :return: generated latent code
        """

        x_data = x_data.reshape(-1, self.input_channels)
        # self.training_phase: is True because I'm using large batch size and it will normalize based on current batch
        generated_code = self.sess.run(self.code_prediction, feed_dict={self.X: x_data, self.training_phase: large_batch}) 

        return generated_code

    def generate(self, x_data):
        """
        Reconstruct input by passing through encoder and decoder

        :param x_data: input data

        :return: generated input data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        # self.training_phase: is True because I'm using large batch size and it will normalize based on current batch
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data, self.training_phase: True})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss

class EncoderDecoderNetwork_dropout:
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
    ):
        """
        Implement an encoder decoder network and train it

        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep')

        self.code_prediction = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))

        # Define optimizer
        with tf.name_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes) + 1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i - 1],
                                          activation=self.activation_fn, name="encoder_" + str(i))
                    net = tf.layers.dropout(inputs=net, rate=self.keep_prob)
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i - 1],
                                          activation=self.activation_fn, name="encoder_" + str(i))
                    net = tf.layers.dropout(inputs=net, rate=self.keep_prob)
            net = tf.layers.dense(inputs=net, units=self.n_dims_code)
            net = tf.layers.dropout(inputs=net, rate=self.keep_prob)
        return net

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i - 1], activation=self.activation_fn,
                                      name="decoder_" + str(i))
                net = tf.layers.dropout(inputs=net, rate=self.keep_prob)
            net = tf.layers.dense(inputs=net, units=self.output_channels, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, logs_path):
        """
        Train using provided data

        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path

        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(TRAINING_EPOCHS):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost],
                                 feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: 0.5})

            cost_log.append(c)

            # Print generated data after every 100 epoch
            # if (epoch + 1) % 100 == 0:
            #     print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(c))
            #     generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})
            #     print("Generated: ")
            #     print(generated_output[0][0:7])
            #     print("Original: ")
            #     print(y_data[0][0:7])

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate(self, x_data):
        """
        Generate target robot data using source robot data

        :param x_data: source robot data

        :return: generated target robot data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data, self.keep_prob: 1.0})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss


class EncoderDecoderNetwork_VAE:
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
            training_epochs=1000,
    ):
        """
        Implement an encoder decoder network and train it

        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.training_epochs = training_epochs

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')

        self.code_prediction, self.z_mu, self.z_log_sigma_sq = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            # Bad reconstruction
            # self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))
            # self.cost = tf.reduce_mean(self.cost)

            # shaohua0116_demo
            # working:
            epsilon = 1e-10
            recon_loss = -tf.reduce_sum(self.Y * tf.log(epsilon+self.output) + (1-self.Y) * tf.log(epsilon+1-self.output), axis=1)
            self.cost = tf.reduce_mean(recon_loss)

            # jmetzen
            # Not working: nans in loss
            # self.cost = -tf.reduce_sum(self.Y * tf.log(1e-10 + self.output) + (1-self.Y) * tf.log(1e-10 + 1 - self.output), 1)

            # takuseno
            # Works
            # entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.output)
            # self.cost = tf.reduce_mean(tf.reduce_sum(entropy, axis=1))

            # LynnHo
            # Bad reconstruction
            # self.cost = tf.losses.mean_squared_error(self.Y, self.output)

            # Latent loss
            # Kullback Leibler divergence: measure the difference between two distributions
            # Here we measure the divergence between the latent distribution and N(0, 1)
            self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
            #print("latent_loss: ", self.latent_loss)
            self.latent_loss = tf.reduce_mean(self.latent_loss)
            #print("latent_loss: ", self.latent_loss)

            self.cost = tf.reduce_mean(self.cost + self.latent_loss)
            #print("cost: ", self.cost)

        # Define optimizer
        with tf.name_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes)+1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_"+str(i))
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_"+str(i))
            #net = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=self.activation_fn)
            z_mu = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=None, name='z_mu')
            z_log_sigma_sq = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=None, name='z_log_sigma_sq')
            eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
            z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps # The reparameterization trick
        return z, z_mu, z_log_sigma_sq

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="decoder_"+str(i))
            # Working:
            net = tf.layers.dense(inputs=net, units=self.output_channels, activation=tf.sigmoid, name="decoder_final") # For MNIST, pixels are between 0 & 1
            #net = tf.layers.dense(inputs=net, units=self.output_channels, activation=self.activation_fn, name="decoder_final")
            # Not Working:
            #net = tf.layers.dense(inputs=net, units=self.output_channels, kernel_initializer=w_init(), name="decoder_final")
            #net = tf.layers.dense(inputs=net, units=self.output_channels, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, logs_path):
        """
        Train using provided data

        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path

        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(self.training_epochs):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict={self.X: x_data, self.Y: y_data})

            cost_log.append(c)

            # Print generated data after every 100 epoch
            # if (epoch + 1) % 100 == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
            #     generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})
            #     print("Generated: ")
            #     print(list(generated_output[0]))
            #     print("Original: ")
            #     print(list(y_data[0]))

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate_code(self, x_data):
        """
        Generate target robot data using source robot data

        :param x_data: source robot data

        :return: generated target robot data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_code = self.sess.run(self.code_prediction, feed_dict={self.X: x_data})

        return generated_code

    def generate(self, x_data):
        """
        Reconstruct input by passing through encoder and decoder

        :param x_data: input data

        :return: generated input data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss

