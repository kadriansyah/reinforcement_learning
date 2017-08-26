import numpy as np
import gym
import random
import itertools
import tensorflow as tf
import time
from collections import deque
from collections import namedtuple

EPISODES = 5000
GAMMA = 0.99
LEARNING_RATE_POLICY_ESTIMATOR = 0.001
LEARNING_RATE_VALUE_ESTIMATOR = 0.01
RENDER = True
SAVE = True
RELOAD = False

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
            self.action = tf.placeholder(dtype=tf.int32)
            self.target = tf.placeholder(tf.float32)

            state_one_hot = tf.one_hot(indices=self.state, depth=self.s_dim)
            self.output = tf.contrib.layers.fully_connected(
                            inputs=tf.expand_dims(state_one_hot, 0),
                            num_outputs=self.a_dim,
                            activation_fn=None,
                            weights_initializer=tf.zeros_initializer)

            self.probabilities = tf.squeeze(tf.nn.softmax(self.output))
            self.predicted_action = tf.gather(self.probabilities, self.action)

            # Loss and train op
            self.loss = -(tf.log(self.predicted_action) * self.target)
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

            state_one_hot = tf.one_hot(indices=self.state, depth=self.s_dim)
            self.output = tf.contrib.layers.fully_connected(
                            inputs=tf.expand_dims(state_one_hot, 0),
                            num_outputs=1,
                            activation_fn=None,
                            weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

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

env = gym.make('CliffWalking-v0')

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
    if RELOAD:
        saver.restore(sess, "save/cliff_walking-ac-baseline/cliff_walking-ac-baseline.ckpt")

    success_episodes = 0
    success_episodes_with_shortest_path = 0
    rewards = []
    for episode in range(0, EPISODES):
        state = env.reset()
        total_reward = 0
        memory = []
        for _ in itertools.count():
            if RENDER:
                env.render()

            # choose action based on policy gradient
            action_probs = policy_estimator.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            # store the transition in D: <state, action, reward, next_state, done>
            memory.append(
                Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

            total_reward += reward
            state = next_state

            if done:
                print("Episode: {}/{}, Total Reward: {}".format(episode + 1, EPISODES, total_reward))
                success_episodes += 1 if reward == -1 else 0
                success_episodes_with_shortest_path += 1 if total_reward == -13 else 0
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

        if SAVE and episode > 0 and (episode + 1) % 1000 == 0:
            saver.save(sess, "save/cliff_walking-ac-baseline/cliff_walking-ac-baseline.ckpt")

        if episode > 0 and (episode + 1) % 1000 == 0:
            print("Episode: {:d}".format(episode + 1))
            print("Episode Rewards: {:.2f}".format(total_reward))
            print("Total Rewards: {:d}".format(sum(rewards)))
            print("Percent of succesful episodes: {:.2f}%".format((success_episodes/EPISODES) * 100))
            print("Percent of successul episodes with shortest path: {:.2f}".format((success_episodes_with_shortest_path/success_episodes) * 100))
            time.sleep(2.5)
