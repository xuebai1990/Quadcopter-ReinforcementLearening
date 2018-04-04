import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
import numpy as np

# Hyperparamters
HIDDEN1_SIZE = 400
HIDDEN2_SIZE = 300

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

        self.state, self.action, self.phase = self.create_network()
        self.tar_state, self.tar_action, self.tar_phase = self.create_target()
        self.create_train()
        self.sess.run(tf.global_variables_initializer())
        self.update_target(True)

    def create_network(self):
        with tf.variable_scope("train"):

            # Placeholders
            state = tf.placeholder(tf.float32, [None, self.state_size])
            phase = tf.placeholder(tf.bool)

            # Define network
            bn1 = self.BatchNorm(state, phase, scope='bn1')
            hidden1 = fully_connected(bn1, HIDDEN1_SIZE, scope='hidden1', activation_fn=None)
            bn2 = self.BatchNorm(hidden1, phase, scope='bn2')
            activate1 = tf.nn.relu(bn2)
            hidden2 = fully_connected(activate1, HIDDEN2_SIZE, scope='hidden2', activation_fn=None)
            bn3 = self.BatchNorm(hidden2, phase, scope='bn3')
            activate2 = tf.nn.relu(bn3)

            # Define action
            raw_action = fully_connected(activate2, self.action_size, activation_fn=tf.nn.sigmoid, scope='raw')
            action = tf.add(tf.multiply(raw_action, self.range), self.action_low)

            return state, action, phase


    def create_target(self):
        with tf.variable_scope("target"):

            # Placeholders
            state = tf.placeholder(tf.float32, [None, self.state_size])
            phase = tf.placeholder(tf.bool)

            # Define network
            bn1 = self.BatchNorm(state, phase, scope='tar_bn1')
            hidden1 = fully_connected(bn1, HIDDEN1_SIZE, scope='tar_hidden1', activation_fn=None)
            bn2 = self.BatchNorm(hidden1, phase, scope='tar_bn2')
            activate1 = tf.nn.relu(bn2)
            hidden2 = fully_connected(activate1, HIDDEN2_SIZE, scope='tar_hidden2', activation_fn=None)
            bn3 = self.BatchNorm(hidden2, phase, scope='tar_bn3')
            activate2 = tf.nn.relu(bn3)

            # Define action
            raw_action = fully_connected(activate2, self.action_size, activation_fn=tf.nn.sigmoid, scope='tar_raw')
            action = tf.add(tf.multiply(raw_action, self.range), self.action_low)

            return state, action, phase

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
        action_gradients = np.squeeze(action_gradients)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "train")
        with tf.control_dependencies(update_ops):
            self.sess.run(self.optimizer, feed_dict={self.action_gradients:action_gradients, self.state:state_batch, self.phase:True})

    def actions(self, state):
        return self.sess.run(self.action, feed_dict={self.state:state, self.phase:False})

    def target_actions(self, state):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "target")
        with tf.control_dependencies(update_ops):
            return self.sess.run(self.tar_action, feed_dict={self.tar_state:state, self.tar_phase:False})

    def variables(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

    def BatchNorm(self, inputT, training, scope=None):
        return tf.cond(training,
                lambda: batch_norm(inputT, is_training=True,
                                   center=True, scale=True, updates_collections=None, scope=scope),
                lambda: batch_norm(inputT, is_training=False,
                                   center=True, scale=True, updates_collections=None, scope=scope, reuse = True))





