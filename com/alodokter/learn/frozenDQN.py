import numpy as np
import gym
import random
import tensorflow as tf
from collections import deque
import time

REPLAY_MEMORY = 1000 # number of previous transitions to remember
BATCH_SIZE = 1 # size of minibatch
EPISODES = 1000
GAMMA = 0.99
EPSILON = 0.1
LEARNING_RATE = 0.1
TIMES = 500
RENDER = True

env = gym.make('FrozenLake-v0')
# env = gym.wrappers.Monitor(env, 'exp_n1')

state_dim  = env.observation_space.n
action_dim = env.action_space.n

tf.reset_default_graph()

inputs = tf.placeholder(shape=[None,state_dim], dtype=tf.float32)
W = tf.get_variable(
        "W",
        initializer=tf.random_uniform((state_dim,action_dim), 0, 0.01))
Qout = tf.matmul(net, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[None,action_dim], dtype=tf.float32)
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

    # # Restore model weights from previously saved model
    # saver.restore(sess, "save/frozen-dqn.ckpt")
    for epoch in range(5):
        rewards = []
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            for time_step in range(TIMES):
                if RENDER:
                    env.render()

                if np.random.rand(1) < EPSILON:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    action = sess.run(predict, feed_dict={inputs:one_hot_encoding(state)})[0]

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                # store the transition in D: <state, action, reward, next_state>
                memory.append((state, action, reward, next_state, done))
                if len(memory) > BATCH_SIZE:
                    memory.popleft()

                # experience replay
                if len(memory) == BATCH_SIZE:
                    # sample a minibatch to train on
                    minibatch = random.sample(memory, BATCH_SIZE)

                    states  = np.zeros((BATCH_SIZE, state_dim)) # states
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
                            Q[i, action_t] = reward_t
                        else:
                            Q_t1 = sess.run(Qout, feed_dict={inputs:one_hot_encoding(state_t1)})
                            Q[i, action_t] = reward_t + GAMMA * np.max(Q_t1)

                    # Train our network using target and predicted Q values
                    _ = sess.run(train_op, feed_dict={inputs:states, nextQ:Q})
                    # memory.clear()

                total_reward += reward
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2f}".format(episode, EPISODES, total_reward, EPSILON))
                    EPSILON = 1.0/((episode/50) + 10) # Reduce chance of random action as we train the model.
                    break

            rewards.append(total_reward)

            if episode > 0 and episode % 100 == 0:
                saver.save(sess, "save/frozen-dqn.ckpt")

        print("\n")
        print("epoch: {:d}".format(epoch))
        print("Rewards: {:.2f}".format(sum(rewards)))
        print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
        time.sleep(2.5)
