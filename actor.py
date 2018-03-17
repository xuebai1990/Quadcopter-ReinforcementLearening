import tensorflow as tf
import numpy as np

# State -> Action
class ActorNet:
    def __init__(self, sess, state_size, action_size, learning_rate, action_low, action_high, tau):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.range = action_high - action_low
        self.learning_rate = learning_rate
        self.tau = tau

        self.create_network()
        self.create_target()
        self.create_train()
        self.sess.run(tf.initialize_all_variables())
        self.update_target(True)

    def create_network(self):
        with tf.variable_scope("train"):

            # Placeholders
            self.state = tf.placeholder(tf.float32, [None, self.state_size])
            self.phase = tf.placeholder(tf.bool)

            # Define network
            self.bn1 = tf.contrib.layers.batch_norm(self.state, center=True, scale=True, is_training=self.phase)
            self.hidden1 = tf.contrib.layers.fully_connected(self.bn1, 32)
            self.bn2 = tf.contrib.layers.batch_norm(self.hidden1, center=True, scale=True, is_training=self.phase)
            self.hidden2 = tf.contrib.layers.fully_connected(self.bn2, 64)
            self.bn3 = tf.contrib.layers.batch_norm(self.hidden2, center=True, scale=True, is_training=self.phase)
            self.hidden3 = tf.contrib.layers.fully_connected(self.bn3, 32)
            self.bn4 = tf.contrib.layers.batch_norm(self.hidden3, center=True, scale=True, is_training=self.phase)

            # Define action
            self.raw_action = tf.contrib.layers.fully_connected(self.bn4, self.action_size, activation_fn=tf.nn.sigmoid)
            self.action = tf.add(tf.multiply(self.raw_action, self.range), self.action_low)


    def create_target(self):
        with tf.variable_scope("target"):
            self.tar_state = tf.placeholder(tf.float32, [None, self.state_size])
            self.tar_phase = tf.placeholder(tf.bool)

            # Define network
            self.tar_bn1 = tf.contrib.layers.batch_norm(self.tar_state, center=True, scale=True, is_training=self.tar_phase)
            self.tar_hidden1 = tf.contrib.layers.fully_connected(self.tar_bn1, 32)
            self.tar_bn2 = tf.contrib.layers.batch_norm(self.tar_hidden1, center=True, scale=True, is_training=self.tar_phase)
            self.tar_hidden2 = tf.contrib.layers.fully_connected(self.tar_bn2, 64)
            self.tar_bn3 = tf.contrib.layers.batch_norm(self.tar_hidden2, center=True, scale=True, is_training=self.tar_phase)
            self.tar_hidden3 = tf.contrib.layers.fully_connected(self.tar_bn3, 32)
            self.tar_bn4 = tf.contrib.layers.batch_norm(self.tar_hidden3, center=True, scale=True, is_training=self.tar_phase)

            # Define action
            self.tar_raw_action = tf.contrib.layers.fully_connected(self.tar_bn4, self.action_size, activation_fn=tf.nn.sigmoid)
            self.tar_action = tf.add(tf.multiply(self.tar_raw_action, self.range), self.action_low)



    def update_target(self, init):
        target_network_update = []
        for v_target, v_source in zip(self.variables("target"), self.variables("train")):
            if init:
                update_op = v_target.assign(v_source)
            else:
                update_op = v_target.assign(v_source * self.tau + v_target * (1 - self.tau))
            target_network_update.append(update_op)
        self.sess.run(tf.group(*target_network_update))


    def create_train(self):
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
        self.loss = tf.reduce_mean(tf.multiply(-self.action_gradients, self.action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, action_gradients, state_batch):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
         self.sess.run(self.optimizer, feed_dict={self.action_gradients = action_gradients, self.state = state_batch, self.phase = True})

    def actions(self, state):
        return self.sess.run(self.action, feed_dict={self.state=state, self.phase=False})

    def target_actions(self, state):
        return self.sess.run(self.tar_action, feed_dict={self.tar_state=state, self.tar_phase=True})

    def variables(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)







