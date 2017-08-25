import numpy as np
import gym
import random
import tensorflow as tf
import time

EPISODES = 50000
GAMMA = 0.99
PRE_TRAIN_STEPS = 0 # Number of steps used before training updates begin.
EPSILON = 0.1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TIMES = 500
RENDER = True

env = gym.make('FrozenLake-v0')
# env = gym.wrappers.Monitor(env, 'exp_n1')

state_dim  = env.observation_space.n
action_dim = env.action_space.n

tf.reset_default_graph()

inputs = tf.placeholder(shape=[1, state_dim], dtype=tf.float32)

W = tf.get_variable(
        "W",
        shape=[state_dim, action_dim],
        initializer=tf.zeros_initializer)

Q_out = tf.matmul(inputs, W)
predict = tf.argmax(Q_out, 1)

Q_next = tf.placeholder(shape=[1, action_dim], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_next - Q_out))
train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def one_hot_encoding(x):
    return np.identity(state_dim)[x:x+1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Restore model weights from previously saved model
    saver.restore(sess, "save/frozen-dqn/frozen-dqn.ckpt")

    total_steps = 0
    rewards = []
    for episode in range(0, EPISODES):
        state = env.reset()
        total_reward = 0
        for _ in range(TIMES):
            if RENDER:
                env.render()

            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) <= EPSILON or total_steps < PRE_TRAIN_STEPS:
                Q = sess.run(Q_out, feed_dict={inputs:one_hot_encoding(state) })
                act = env.action_space.sample()
            else:
                action, Q = sess.run([predict, Q_out], feed_dict={inputs:one_hot_encoding(state) })
                act = action[0]

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(act)

            if done:
                r = 1 if reward > 0 else -1
            else:
                r = 0

            Q_next_state = sess.run(Q_out, feed_dict={inputs:one_hot_encoding(next_state) })
            Q[0, act] = r + GAMMA * np.max(Q_next_state)

            # train the model
            sess.run(train_op, feed_dict={inputs:one_hot_encoding(state), Q_next:Q })

            total_reward += reward
            state = next_state
            total_steps += 1
            if done:
                print("episode: {}/{}, score: {}, e: {:.2f}".format(episode + 1, EPISODES, total_reward, EPSILON))
                if total_steps > PRE_TRAIN_STEPS:
                    EPSILON *= EPSILON_DECAY if EPSILON > EPSILON_MIN else 1
                break

        rewards.append(total_reward)

        if episode > 0 and (episode + 1) % 500 == 0:
            saver.save(sess, "save/frozen-dqn/frozen-dqn.ckpt")

        if episode > 0 and (episode + 1) % 1000 == 0:
            print("episode: {:d}".format(episode + 1))
            print("Rewards: {:.2f}".format(sum(rewards)))
            print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
            time.sleep(2.5)
