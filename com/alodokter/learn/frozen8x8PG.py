import numpy as np
import gym
import random
import tensorflow as tf
import time
from collections import deque
from collections import namedtuple

EPISODES = 10000
GAMMA = 0.99
LEARNING_RATE_POLICY_ESTIMATOR = 0.01
LEARNING_RATE_VALUE_ESTIMATOR = 0.01
TIMES = 500
RENDER = True

class PolicyEstimator(object):
    """
    Policy Function approximator.
    """
    def __init__(self, sess, s_dim, a_dim, learning_rate=0.001, scope="policy_estimator"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        with tf.variable_scope(scope):
            self.state  = tf.placeholder(shape=[], dtype=tf.int32)
            self.action = tf.placeholder(shape=[1, self.a_dim], dtype=tf.int32)
            self.target = tf.placeholder(tf.float32)

            state_one_hot = tf.one_hot(self.state, int(self.s_dim))
            self.output = tf.contrib.layers.fully_connected(
                            inputs=tf.expand_dims(state_one_hot, 0),
                            num_outputs=self.a_dim,
                            activation_fn=None,
                            weights_initializer=tf.zeros_initializer)

            self.probabilities = tf.squeeze(tf.nn.softmax(self.output))
            self.predicted_action = tf.gather(self.probabilities, self.action)

            # Loss and train op
            self.loss = -tf.log(self.predicted_action) * self.target
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        return self.sess.run(self.probabilities, feed_dict={self.state: state})

    def update(self, state, target, action):
        _, loss = self.sess.run(
                    [self.train_op, self.loss],
                    feed_dict={self.state:  state,self.target: target,self.action: action })
        return loss

class ValueEstimator(object):
    """
    Value Function approximator.
    """
    def __init__(self, sess, s_dim, a_dim, learning_rate=0.01, scope="value_estimator"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        with tf.variable_scope(scope):
            self.state  = tf.placeholder(shape=[],  dtype=tf.int32)
            self.target = tf.placeholder(tf.float32)

            state_one_hot = tf.one_hot(self.state, int(self.s_dim))
            self.output = tf.contrib.layers.fully_connected(
                            inputs=tf.expand_dims(state_one_hot, 0),
                            num_outputs=1,
                            activation_fn=None,
                            weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output)
            # self.loss = tf.squared_difference(self.value_estimate, self.target)
            self.loss = tf.reduce_sum(tf.square(self.target - self.value_estimate))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        return self.sess.run(self.value_estimate, feed_dict={ self.state: state })

    def update(self, state, target):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.state:  state, self.target: target })
        return loss

def xrange(x):
    return iter(range(x))

def xrange(s,x):
    return range(s,x)

def one_hot_encoding(dim, x):
    return np.identity(dim)[x:x+1]

env = gym.make('FrozenLake8x8-v0')
# env = gym.wrappers.Monitor(env, 'exp_n1')

s_dim = env.observation_space.n
a_dim = env.action_space.n

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

with tf.Session() as sess:

    policy_estimator = PolicyEstimator(sess, s_dim, a_dim, LEARNING_RATE_POLICY_ESTIMATOR)
    value_estimator = ValueEstimator(sess, s_dim, a_dim, LEARNING_RATE_VALUE_ESTIMATOR)

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    # store the previous observations in replay memory
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # Restore model weights from previously saved model
    saver.restore(sess, "save/frozen8x8-pg/frozen8x8-pg.ckpt")

    total_steps = 0
    rewards = []
    for episode in range(0, EPISODES):
        state = env.reset()
        total_reward = 0
        memory = []
        for _ in range(TIMES):
            if RENDER:
                env.render()

            # choose action based on policy gradient
            action_probs = policy_estimator.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            if done:
                r = 10 if reward > 0 else -10
            else:
                r = 0

            # store the transition in D: <state, action, reward, next_state, done>
            memory.append(
                Transition(state=state, action=one_hot_encoding(a_dim, action), reward=r, next_state=next_state, done=done))

            total_reward += reward
            state = next_state
            total_steps += 1

            if done:
                print("episode: {}/{}, score: {}".format(episode + 1, EPISODES, total_reward))
                break

        rewards.append(total_reward)

        # go through the episode and make policy updates
        for t, transition in enumerate(memory):
            total_return = sum(GAMMA**i * t.reward for i, t in enumerate(memory[t:]))

            # update our value estimator
            value_estimator.update(transition.state, total_return)

            # calculate baseline/advantage
            baseline_value = value_estimator.predict(transition.state)
            advantage = total_return - baseline_value

            # update our policy estimator
            policy_estimator.update(transition.state, advantage, transition.action)

        if episode > 0 and (episode + 1) % 1000 == 0:
            saver.save(sess, "save/frozen8x8-pg/frozen8x8-pg.ckpt")

        if episode > 0 and (episode + 1) % 1000 == 0:
            print("episode: {:d}".format(episode + 1))
            print("Rewards: {:.2f}".format(sum(rewards)))
            print("Percent of succesful episodes: {:.2f}%".format((sum(rewards)/episode) * 100))
            time.sleep(2.5)
