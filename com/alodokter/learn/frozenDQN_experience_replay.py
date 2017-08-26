import numpy as np
import gym
import random
import tensorflow as tf
from collections import deque
import time

REPLAY_MEMORY = 128 # number of previous transitions to remember
BATCH_SIZE = 16 # size of minibatch
EPISODES = 3000
GAMMA = 0.99
EPSILON = 0.1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.1
TIMES = 500
RENDER = True
HIDDEN_SIZE = 64

env = gym.make('FrozenLake-v0')

state_dim  = env.observation_space.n
action_dim = env.action_space.n

tf.reset_default_graph()

inputs = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
W1 = tf.get_variable(
        "W1",
        shape=[state_dim, HIDDEN_SIZE],
        initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(inputs,W1))
W2 = tf.get_variable(
        "W2",
        shape=[HIDDEN_SIZE, action_dim],
        initializer=tf.contrib.layers.xavier_initializer())
Qout = tf.matmul(layer1,W2)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[None, action_dim], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def one_hot_encoding(x):
    return np.identity(state_dim)[x:x+1]

with tf.Session() as sess:
    sess.run(init)

    # store the previous observations in replay memory
    memory = deque(maxlen=REPLAY_MEMORY)

    # Restore model weights from previously saved model
    saver.restore(sess, "save/frozen-dqn/frozen-dqn.ckpt")
    rewards = []
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        for _ in range(TIMES):
            if RENDER:
                env.render()

            # e-greedy approach
            if np.random.rand(1) < EPSILON:
                action = env.action_space.sample()
            else:
                # Choose an action by greedily (with e chance of random action) from the Q-network
                action = sess.run(predict, feed_dict={inputs:one_hot_encoding(state)})[0]

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            # store the transition in D: <state, action, reward, next_state>
            memory.append((state, action, reward, next_state, done))
            if len(memory) > REPLAY_MEMORY:
                memory.popleft()

            total_reward += reward
            state = next_state
            if done:
                # experience replay
                if len(memory) >= BATCH_SIZE:
                    # sample a minibatch to train on
                    minibatch = random.sample(memory, BATCH_SIZE)

                    states = np.zeros((BATCH_SIZE, state_dim)) # states
                    Q = np.zeros((BATCH_SIZE, action_dim)) # actions score
                    for i in range(0, len(minibatch)):
                        state_t  = minibatch[i][0]
                        action_t = minibatch[i][1]
                        reward_t = minibatch[i][2]
                        state_t1 = minibatch[i][3]
                        done     = minibatch[i][4]

                        states[i] = one_hot_encoding(state_t)
                        Q[i] = sess.run(Qout, feed_dict={inputs:one_hot_encoding(state_t)})
                        if done:
                            Q[i, action_t] = 1 if reward_t > 0 else -1 # give penalty
                        else:
                            Q_t1 = sess.run(Qout, feed_dict={inputs:one_hot_encoding(state_t1)})
                            Q[i, action_t] = reward_t + GAMMA * np.max(Q_t1)

                    # Train our network using target and predicted Q values
                    sess.run(train_op, feed_dict={inputs:states, nextQ:Q})

                print("episode: {}/{}, score: {}, e: {:.2f}".format(episode, EPISODES, total_reward, EPSILON))
                # Reduce chance of random action as we train the model.
                EPSILON *= EPSILON_DECAY if EPSILON > EPSILON_MIN else 1
                break

        rewards.append(total_reward)
        if episode > 0 and episode % 500 == 0:
            saver.save(sess, "save/frozen-dqn/frozen-dqn.ckpt")

        if episode > 0 and episode % 1000 == 0:
            print("episode: {:d}".format(episode))
            print("Rewards: {:.2f}".format(sum(rewards)))
            print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
            time.sleep(2.5)
