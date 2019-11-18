import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.ERROR)

import os
import numpy as np

class Model(object):
    def __init__(self, observation_size, env):
        self.batch_size = 32
        self.observation_size = observation_size
        self.gamma = 0.005
        self.learning_rate = 0.0005
        self.num_nodes = [200, 100, 1]  # 1 output for probability of going up
        self.env = env

        self.sess = tf.Session()

        self.check_point_dir = './models'
        self.model_name = 'deepQnet.ckpt'

        self.current_step = 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.training = False

        self.inputs()

        # q_primary decides action for current state
        self.q_primary = self.buildModel(input=self.states, name='Q_primary')

        self.loss_acc()

        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(tf.global_variables_initializer())

        # self.load_model()

    def inputs(self):
        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            # shape = [batch size, observation size]
            self.states = tf.placeholder(tf.float32, shape=(None, self.observation_size), name='states')
            self.actions = tf.placeholder(tf.float32, shape=(None, 1), name='actions')
            self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')

    def loss_acc(self):
        with tf.variable_scope('loss_acc', reuse=tf.AUTO_REUSE):
            '''
            Main idea: 
            Decide sample actions by their outcome rewards, if outcome is bad (reward < 0.),
            discourage the action by multiplying action with reward, otherwise action is encouraged.
            
            '''

            # self.loss = tf.losses.log_loss(
            #     labels=self.actions,
            #     predictions=self.q_primary,
            #     weights=self.rewards,
            # )
            # self.loss = tf.reduce_mean(self.log_loss)
            # self.loss = tf.reduce_mean(tf.square(self.q_primary - tf.multiply(self.rewards, self.actions)))

            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.multiply(self.rewards, self.actions),
                logits=self.q_primary
            )
            self.reduced_loss = tf.reduce_mean(self.loss)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)\
                .minimize(self.reduced_loss, name='adam_optimization', global_step=self.global_step)


    def buildModel(self, input, name):

        with tf.variable_scope('{}'.format(name), reuse=tf.AUTO_REUSE):
            # # fc layer 1
            # fc1 = tf.layers.dense(input,
            #                       self.num_nodes[0],
            #                       activation=tf.nn.relu,
            #                       kernel_initializer=tf.keras.initializers.glorot_uniform(),
            #                       name='{}_fc_1'.format(name))
            #
            # # fc layer 2
            # fc2 = tf.layers.dense(fc1,
            #                       self.num_nodes[1],
            #                       activation=tf.nn.relu,
            #                       kernel_initializer=tf.keras.initializers.glorot_uniform(),
            #                       name='{}_fc_2'.format(name))

            # fc layer 2
            fc2 = tf.layers.dense(input,
                                  self.num_nodes[0],
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.keras.initializers.glorot_normal(),
                                  name='{}_fc_2'.format(name))

            # output layer
            output = tf.layers.dense(fc2,
                                     self.num_nodes[2],
                                     activation=None,
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                     name='{}_output'.format(name))

            return output

    def save_model(self, steps):
        print('Saving checkpoints...')
        ckpt_file = os.path.join(self.check_point_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=self.global_step)

    def load_model(self):
        print('Loading checkpoints...')
        ckpt_path = tf.train.latest_checkpoint(self.check_point_dir)
        print('checkpoint dir: {}'.format(ckpt_path))

        if ckpt_path:
            self.saver.restore(self.sess, ckpt_path)
            print('Load model success!')

            self.current_step = self.global_step.eval(session=self.sess)
            print('Model restored at step {}'.format(self.current_step))
            return True
        else:
            print('Load model failure')
            return False

    def scope_vars(self, name):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES
        variables = tf.get_collection(collection, scope=name)
        return variables

    def observable_to_input(self, state):
        return state.flatten()

    def return_action(self, state, epsilon=0.1):
        up_probability = tf.sigmoid(self.q_primary).eval({self.states: state.reshape(1, -1)})[0]
        # up_probability = self.q_primary.eval({self.states: state.reshape(1, -1)})[0]
        # print('up probability: {}'.format(up_probability))

        if np.random.uniform() < up_probability:
            return 2
        else:
            return 3

