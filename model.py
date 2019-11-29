import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.ERROR)

import os
import numpy as np

class Model(object):
    def __init__(self, observation_size, env):
        self.observation_size = observation_size
        self.gamma = 0.005
        self.learning_rate = 0.0005
        self.num_nodes = [200, 1]  # 1 output for probability of going up
        self.env = env

        self.sess = tf.Session()

        self.check_point_dir = './models'
        self.model_name = 'deepQnet.ckpt'

        self.current_step = 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.training = False

        self.inputs()

        # q_primary decides action for current state
        self.network = self.buildModel(input=self.states, name='network')

        self.loss_acc()

        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(tf.global_variables_initializer())

        self.load_model()

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
            #     predictions=self.network,
            #     weights=self.rewards
            # )

            self.epsilon = 1e-7
            self.loss = tf.reduce_mean(
                (self.rewards * self.actions * -tf.log(self.network + self.epsilon)+
                 (self.rewards * (1. - self.actions) * -tf.log(1. - self.network + self.epsilon)))
            )

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, name='adam_optimization', global_step=self.global_step)


    def buildModel(self, input, name):

        with tf.variable_scope('{}'.format(name), reuse=tf.AUTO_REUSE):
            # fc layer 2
            fc2 = tf.layers.dense(input,
                                  self.num_nodes[0],
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.keras.initializers.glorot_normal(),
                                  name='{}_fc_2'.format(name))

            # output layer
            output = tf.layers.dense(fc2,
                                     self.num_nodes[-1],
                                     activation=tf.nn.sigmoid,
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
        up_probability = self.network.eval({self.states: state.reshape(1, -1)})[0]

        if np.random.uniform() < up_probability:
            return 2
        else:
            return 3

