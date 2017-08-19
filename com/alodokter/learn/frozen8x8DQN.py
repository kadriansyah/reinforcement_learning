import numpy as np
import gym
import random
import tensorflow as tf
import time

EXPLORATION = 'e-greedy' # Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.
EPISODES = 3000
GAMMA = 0.99
ANNELING_STEPS  = 3000 # How many steps of training to reduce START_EPSILON to END_EPSILON.
PRE_TRAIN_STEPS = 0 # Number of steps used before training updates begin.
EPSILON = 0.01
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.01
TIMES = 500
RENDER = True

env = gym.make('FrozenLake8x8-v0')
# env = gym.wrappers.Monitor(env, 'exp_n1')

state_dim  = env.observation_space.n
action_dim = env.action_space.n

tf.reset_default_graph()

inputs = tf.placeholder(shape=[1, state_dim], dtype=tf.float32)
temp = tf.placeholder(tf.float32)
keep = tf.placeholder(tf.float32)

hidden = tf.contrib.layers.fully_connected(inputs, 64, activation_fn=tf.nn.tanh, biases_initializer=None)
hidden = tf.nn.dropout(hidden, keep)
Q_out = tf.contrib.layers.fully_connected(hidden, action_dim , activation_fn=None, biases_initializer=None)

predict = tf.argmax(Q_out, 1)
Q_dist = tf.nn.softmax(Q_out/temp)

Q_next = tf.placeholder(shape=[1, action_dim], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_next - Q_out))
train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def one_hot_encoding(x):
    return np.identity(state_dim)[x:x+1]

with tf.Session() as sess:
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, "save/frozen8x8-dqn/frozen8x8-dqn.ckpt")

    total_steps = 0
    rewards = []
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        for _ in range(TIMES):
            if RENDER:
                env.render()

            if EXPLORATION == "greedy":
                # Choose an action with the maximum expected value.
                action, Q = sess.run([predict,Q_out], feed_dict={inputs:one_hot_encoding(state), keep:1.0 })
                act = action[0]

            elif EXPLORATION == "random":
                Q = sess.run(Q_out, feed_dict={inputs:one_hot_encoding(state), keep:1.0 })
                act =  env.action_space.sample()

            elif EXPLORATION == "e-greedy":
                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand() <= EPSILON or total_steps < PRE_TRAIN_STEPS:
                    Q = sess.run(Q_out, feed_dict={inputs:one_hot_encoding(state), keep:1.0 })
                    act = env.action_space.sample()
                else:
                    action, Q = sess.run([predict, Q_out], feed_dict={inputs:one_hot_encoding(state), keep:1.0 })
                    act = action[0]

            elif EXPLORATION == "boltzmann":
                # Choose an action probabilistically, with weights relative to the Q-values.
                Qdist, Q = sess.run([Q_dist, Q_out],feed_dict={ inputs:one_hot_encoding(state), temp:EPSILON , keep:1.0 })
                act = np.random.choice(Qdist[0], p=Qdist[0])
                act = np.argmax(Qdist[0] == act)

            elif EXPLORATION == "bayesian":
                action, Q = sess.run([predict, Q_out],feed_dict={ inputs:one_hot_encoding(state), keep:(1-EPSILON)+0.1 })
                act = action[0]

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(act)

            if done:
                r = 1 if reward > 0 else -1 # give penalty
            else:
                r = 0

            Q_next_state = sess.run(Q_out, feed_dict={inputs:one_hot_encoding(next_state), keep:1.0 })
            Q[0, act] = r + GAMMA * np.max(Q_next_state)

            # train the model
            sess.run(train_op, feed_dict={inputs:one_hot_encoding(state), Q_next:Q, keep:1.0 })

            total_reward += reward
            state = next_state
            total_steps += 1
            if done:
                print("episode: {}/{}, score: {}, e: {:.2f}".format(episode, EPISODES, total_reward, EPSILON))
                EPSILON *= EPSILON_DECAY if EPSILON > EPSILON_MIN else 1
                break

        rewards.append(total_reward)

        if episode > 0 and episode % 500 == 0:
            saver.save(sess, "save/frozen8x8-dqn/frozen8x8-dqn.ckpt")

        if episode > 0 and episode % 1000 == 0:
            print("episode: {:d}".format(episode))
            print("Rewards: {:.2f}".format(sum(rewards)))
            print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
            time.sleep(2.5)
