import tensorflow as tf

class EncoderDecoderNetwork_b_VAE:
    def __init__(
            self,
            num_of_domains,
            num_of_features,
            domain_names,
            activation_fn,
            beta = 1,
            hidden_layer_sizes=[1000, 500, 250],
            learning_rate=0.0001,
            training_epochs=1000,
    ):
        """
        Implement an beta auto encoder network and train it

        :param num_of_domains: number of domains
        :param num_of_features: a list of number of features in each domain
        :param domain_names: domain names
        :param activation_fn: activation function
        :param beta: beta
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param training_epochs: training epochs
        """

        self.num_of_domains = num_of_domains
        self.num_of_features = num_of_features
        self.domain_names = domain_names
        self.activation_fn = activation_fn
        self.domain_names = domain_names
        self.beta = beta
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs

        self.placeholder = {"input":[], "output":[], "prediction": []}

        for a_domain in range(self.num_of_domains):
            self.placeholder["input"].append(tf.placeholder("float", [None, self.num_of_features[a_domain]], name='input_'+str(a_domain)))
            self.placeholder["output"].append(tf.placeholder("float", [None, self.num_of_features[a_domain]], name='output_'+str(a_domain)))

        encoder_outputs = self.encoder(self.placeholder["input"][0], self.domain_names[0])
        for a_domain in range(1, self.num_of_domains):
            a_encoder_output = self.encoder(self.placeholder["input"][a_domain], self.domain_names[a_domain])
            encoder_outputs= tf.concat([encoder_outputs, a_encoder_output], axis=1, name='encoder_outputs')
        
        self.code_prediction, self.z_mu, self.z_log_sigma_sq = self.latent_code(encoder_outputs)
        
        for a_domain in range(self.num_of_domains):
            a_decoder_output = self.decoder(self.code_prediction, self.num_of_features[a_domain], self.domain_names[a_domain])
            self.placeholder["prediction"].append(a_decoder_output)
        
        # Reconstruction cost
        # concatenating all the domains to optimize them together
        prediction_concat = self.placeholder["prediction"][0]
        output_concat = self.placeholder["output"][0]
        for a_domain in range(1, self.num_of_domains):
            prediction_concat = tf.concat([prediction_concat, self.placeholder["prediction"][a_domain]], axis=1, name='prediction_concat')
            output_concat = tf.concat([output_concat, self.placeholder["output"][a_domain]], axis=1, name='output_concat')
        
        recon_loss = -tf.reduce_sum(output_concat * tf.log(1e-10+prediction_concat) + (1-output_concat) * tf.log(1e-10+1-prediction_concat), axis=1)
        recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        kl_penalty = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        kl_penalty = tf.reduce_mean(kl_penalty)

        self.cost = tf.reduce_mean(recon_loss + self.beta * kl_penalty)
        
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


    def encoder(self, X, domain_name):
        for i in range(1, len(self.hidden_layer_sizes)+1):
            if i == 1:
                net = tf.layers.dense(inputs=X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="Encoder_"+domain_name+"_layer_"+str(i))
                #print(net)
            else:
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="Encoder_"+domain_name+"_layer_"+str(i))
                #print(net)
        return net

    def latent_code(self, X):
        _, encoder_len = X.get_shape()
        z_mu = tf.layers.dense(inputs=X, units=encoder_len, activation=None, name='z_mu')
        z_log_sigma_sq = tf.layers.dense(inputs=X, units=encoder_len, activation=None, name='z_log_sigma_sq')
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps # The reparameterization trick
        
        return z, z_mu, z_log_sigma_sq

    def decoder(self, net, domain_size, domain_name):
        for i in range(len(self.hidden_layer_sizes), 0, -1):
            net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="Decoder_"+domain_name+"_layer_"+str(i))
            #print(net)
        net = tf.layers.dense(inputs=net, units=domain_size, activation=tf.sigmoid, name="Decoder_Final_"+domain_name) # For MNIST, pixels are between 0 & 1
        #print(net)
        return net

    def train_session(self, feed_dict_train, logs_path):
        """
        Train using provided data

        :param feed_dict_train: feed_dict_train
        :param logs_path: log path

        :return: cost over training
        """

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(self.training_epochs):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict=feed_dict_train)

            cost_log.append(c)

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log



