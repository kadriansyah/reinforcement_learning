import numpy as np
import gym
import random
import itertools
import tensorflow as tf
import time

EPISODES = 5000
GAMMA = 1
LEARNING_RATE_POLICY_ESTIMATOR = 0.01
LEARNING_RATE_VALUE_ESTIMATOR = 0.01
RENDER = True
SAVE = True
RELOAD = True

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
            self.target = tf.placeholder(dtype=tf.float32)

            state_one_hot = tf.one_hot(indices=self.state, depth=self.s_dim)
            self.output = tf.contrib.layers.fully_connected(
                            inputs=tf.expand_dims(state_one_hot, 0),
                            num_outputs=self.a_dim,
                            activation_fn=None,
                            weights_initializer=tf.zeros_initializer)

            self.probabilities = tf.squeeze(tf.nn.softmax(self.output))
            self.chosen_action = tf.gather(self.probabilities, self.action)

            # Loss and train op
            self.loss = -(tf.log(self.chosen_action) * self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        return self.sess.run(self.probabilities, feed_dict={self.state: state})

    def update(self, state, target, action):
        _, loss = self.sess.run(
                    [self.train_op, self.loss],
                    feed_dict={self.state:  state, self.target: target, self.action: action })
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

    # Restore model weights from previously saved model
    if RELOAD:
        saver.restore(sess, "save/cliff_walking-ac/cliff_walking-ac.ckpt")

    success_episodes = 0
    success_episodes_with_shortest_path = 0
    rewards = []
    for episode in range(0, EPISODES):
        state = env.reset()
        total_reward = 0
        for _ in itertools.count():
            if RENDER:
                env.render()

            # choose action based on policy gradient
            action_probs = policy_estimator.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            # Calculate TD Target
            curr_value = value_estimator.predict(state)
            value_next = value_estimator.predict(next_state)
            td_target = reward + GAMMA * value_next
            td_error  = td_target - curr_value

            # Update the value estimator
            value_estimator.update(state, td_target)

            # Update the policy estimator using the td error as our advantage estimate
            policy_estimator.update(state, td_error, action)

            total_reward += reward
            state = next_state

            if done:
                print("Episode: {}/{}, Total Reward: {}".format(episode + 1, EPISODES, total_reward))
                success_episodes += 1 if reward == -1 else 0
                success_episodes_with_shortest_path += 1 if total_reward == -13 else 0
                break

        rewards.append(total_reward)
        if SAVE and episode > 0 and (episode + 1) % 1000 == 0:
            saver.save(sess, "save/cliff_walking-ac/cliff_walking-ac.ckpt")

        if episode > 0 and (episode + 1) % 1000 == 0:
            print("Episode: {:d}".format(episode + 1))
            print("Episode Rewards: {:.2f}".format(total_reward))
            print("Total Rewards: {:d}".format(sum(rewards)))
            print("Percent of succesful episodes: {:.2f}%".format((success_episodes/EPISODES) * 100))
            print("Percent of successul episodes with shortest path: {:.2f}%".format((success_episodes_with_shortest_path/success_episodes) * 100))
            time.sleep(2.5)
