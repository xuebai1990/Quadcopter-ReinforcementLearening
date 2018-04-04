import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import l2_regularizer
import numpy as np

#Hyperparameters
HIDDEN1_SIZE = 400
HIDDEN2_SIZE = 300

class CriticNet:
    def __init__(self, sess, state_size, action_size, learning_rate, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.learning_rate = learning_rate
        self.tau = tau

        self.state, self.action, self.phase, self.q_values = self.create_q_net()
        self.tar_state, self.tar_action, self.tar_phase, self.tar_q_values = self.create_target_q_net()
        self.create_train()
        self.sess.run(tf.global_variables_initializer())
        self.update_target(True)

    def create_q_net(self):
        state = tf.placeholder(tf.float32, [None, self.state_size])
        action = tf.placeholder(tf.float32, [None, self.action_size])
        phase = tf.placeholder(tf.bool)

        with tf.variable_scope("train"):
            # State
            bn1_state = self.BatchNorm(state, phase, scope="bn1_state")
            hidden1_state = fully_connected(bn1_state, HIDDEN1_SIZE, scope='h1_state', activation_fn=None, weights_regularizer=l2_regularizer(0.01))
            bn2_state = self.BatchNorm(hidden1_state, phase, scope="bn2_state")
            activate1_state = tf.nn.relu(bn2_state)
            hidden2_state = fully_connected(activate1_state, HIDDEN2_SIZE, scope='h2_state', activation_fn=None, weights_regularizer=l2_regularizer(0.01))

            # Action
            bn1_action = self.BatchNorm(action, phase, scope="bn1_action")
            hidden1_action = fully_connected(bn1_action, HIDDEN1_SIZE, scope='h1_action', activation_fn=None, weights_regularizer=l2_regularizer(0.01))
            bn2_action = self.BatchNorm(hidden1_action, phase, scope="bn2_action")
            activate2_action = tf.nn.relu(bn2_action)
            hidden2_action = fully_connected(activate2_action, HIDDEN2_SIZE, scope='h2_action', activation_fn=None, weights_regularizer=l2_regularizer(0.01))

            # Concate
            combine = tf.concat([hidden2_state, hidden2_action], axis=-1)
            bn3 = self.BatchNorm(combine, phase, scope="bn3_critic")
            hidden3 = tf.nn.relu(bn3)
            q_values = fully_connected(hidden3, 1, activation_fn=None, scope='q', weights_regularizer=l2_regularizer(0.01))

        return state, action, phase, q_values

    def create_target_q_net(self):
        state = tf.placeholder(tf.float32, [None, self.state_size])
        action = tf.placeholder(tf.float32, [None, self.action_size])
        phase = tf.placeholder(tf.bool)

        with tf.variable_scope("target"):
            # State
            bn1_state = self.BatchNorm(state, phase, scope="tar_bn1_state")
            hidden1_state = fully_connected(bn1_state, HIDDEN1_SIZE, scope='tar_h1_state', activation_fn=None, weights_regularizer=l2_regularizer(0.01))
            bn2_state = self.BatchNorm(hidden1_state, phase, scope="tar_bn2_state")
            activate1_state = tf.nn.relu(bn2_state)
            hidden2_state = fully_connected(activate1_state, HIDDEN2_SIZE, scope='tar_h2_state', activation_fn=None, weights_regularizer=l2_regularizer(0.01))

            # Action
            bn1_action = self.BatchNorm(action, phase, scope="tar_bn1_action")
            hidden1_action = fully_connected(bn1_action, HIDDEN1_SIZE, scope='tar_h1_action', activation_fn=None, weights_regularizer=l2_regularizer(0.01))
            bn2_action = self.BatchNorm(hidden1_action, phase, scope="tar_bn2_action")
            activate2_action = tf.nn.relu(bn2_action)
            hidden2_action = fully_connected(activate2_action, HIDDEN2_SIZE, scope='tar_h2_action', activation_fn=None, weights_regularizer=l2_regularizer(0.01))

            # Concate
            combine = tf.concat([hidden2_state, hidden2_action], axis=-1)
            bn3 = self.BatchNorm(combine, phase, scope="tar_bn3_critic")
            hidden3 = tf.nn.relu(bn3)
            q_values = fully_connected(hidden3, 1, activation_fn=None, scope='tar_q', weights_regularizer=l2_regularizer(0.01))

        return state, action, phase, q_values

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
        self.target_q = tf.placeholder(tf.float32, [None])
        self.loss = tf.reduce_mean(tf.square(self.q_values - self.target_q))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.action_gradients = tf.gradients(self.q_values, self.action)

    def train(self, batch_state, batch_action,target_q):
        target_q = np.squeeze(target_q)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.sess.run(self.optimizer, feed_dict={self.state:batch_state, self.action:batch_action, self.phase:True, self.target_q:target_q})

    def gradients(self, batch_state, batch_action):
        return self.sess.run(self.action_gradients, feed_dict={self.state:batch_state, self.action:batch_action, self.phase:False})

    def Qvalue(self, batch_state, batch_action):
        return self.sess.run(self.q_values, feed_dict={self.state:batch_state, self.action:batch_action, self.phase:False})

    def targetQ(self, batch_state, batch_action):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return self.sess.run(self.tar_q_values, feed_dict={self.tar_state:batch_state, self.tar_action:batch_action, self.tar_phase:False})

    def variables(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

    def BatchNorm(self, inputT, training, scope=None):
        return tf.cond(training,
                lambda: batch_norm(inputT, is_training=True,
                                   center=True, scale=True, updates_collections=None, scope=scope),
                lambda: batch_norm(inputT, is_training=False,
                                   center=True, scale=True, updates_collections=None, scope=scope, reuse = True))


