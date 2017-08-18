import numpy as np
import gym
import random
import tensorflow as tf
import time

EPISODES = 2000
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.01
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
        initializer=tf.contrib.layers.xavier_initializer())
Qout = tf.matmul(inputs, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, action_dim], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def one_hot_encoding(x):
    return np.identity(state_dim)[x:x+1]

with tf.Session() as sess:
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, "save/frozen-dqn-ol.ckpt")
    for epoch in range(2):
        rewards = []
        for episode in range(EPISODES):
            state = env.reset()
            total_reward = 0
            for time_step in range(TIMES):
                if RENDER:
                    env.render()

                action, Q = sess.run([predict,Qout], feed_dict={inputs:one_hot_encoding(state)})
                if np.random.rand() <= EPSILON:
                    action[0] = env.action_space.sample()

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action[0])

                if done:
                    r = 1 if reward > 0 else -1 # give penalty
                else:
                    r = 0

                Qt = sess.run(Qout, feed_dict={inputs:one_hot_encoding(next_state)})
                Q[0, action[0]] = r + GAMMA * np.max(Qt)

                # update model
                sess.run(train_op, feed_dict={inputs:one_hot_encoding(state), nextQ:Q})

                total_reward += reward
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2f}".format(episode, EPISODES, total_reward, EPSILON))
                    # EPSILON = 1.0/((episode/50) + 10) # Reduce chance of random action as we train the model.
                    if EPSILON > EPSILON_MIN:
                        EPSILON *= EPSILON_DECAY
                    break

            rewards.append(total_reward)
            # if EPSILON > EPSILON_MIN:
            #     EPSILON *= EPSILON_DECAY

            if episode > 0 and episode % 100 == 0:
                saver.save(sess, "save/frozen-dqn-ol.ckpt")

        print("\n")
        print("epoch: {:d}".format(epoch))
        print("Rewards: {:.2f}".format(sum(rewards)))
        print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
        time.sleep(2.5)
