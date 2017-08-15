import numpy as np
import gym
import random
import tensorflow as tf
from collections import deque

REPLAY_MEMORY = 1000 # number of previous transitions to remember
BATCH_SIZE = 16 # size of minibatch
EPISODES = 2000
TIME_TO_TRAIN = 10
GAMMA = 0.95
EPSILON = 0.1

env = gym.make('FrozenLake-v0')
# env = gym.wrappers.Monitor(env, 'exp_n1')

tf.reset_default_graph()
inputs = tf.placeholder(shape=[None,16], dtype=tf.float32)
W = tf.get_variable("W", initializer=tf.random_uniform([16,4], 0, 0.01))

Qout = tf.matmul(inputs, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[None,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = trainer.minimize(loss)

init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

rewards = []
with tf.Session() as sess:
    sess.run(init)

    # store the previous observations in replay memory
    memory = deque(maxlen=REPLAY_MEMORY)

    # Restore model weights from previously saved model
    saver.restore(sess, "save/frozen-dqn.ckpt")
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        for time in range(500):
            # env.render()
            # Choose an action by greedily (with e chance of random action) from the Q-network
            action, targetQ = sess.run([predict, Qout], feed_dict={inputs:np.identity(16)[state:state+1]})
            if np.random.rand() < EPSILON:
                action[0] = env.action_space.sample()

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action[0])

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs:np.identity(16)[next_state:next_state+1]})

            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ[0, action[0]] = reward + GAMMA * maxQ1

            # store the transition in D: <state, action, reward, next_state>
            memory.append((state, action[0], reward, next_state, done))
            if len(memory) > REPLAY_MEMORY:
                memory.popleft()

            # experience replay
            if len(memory) > BATCH_SIZE:
                # sample a minibatch to train on
                minibatch = random.sample(memory, BATCH_SIZE)

                inputs_train  = np.zeros((BATCH_SIZE, 16))
                targets = np.zeros((BATCH_SIZE, 4)) # action

                # Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    done = minibatch[i][4]

                    inputs_train[i] = np.identity(16)[state_t:state_t+1]
                    action, targetQ = sess.run([predict, Qout], feed_dict={inputs:np.identity(16)[state_t:state_t+1]})
                    Q1 = sess.run(Qout, feed_dict={inputs:np.identity(16)[state_t1:state_t1+1]})
                    if not done:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q1)
                    else:
                        targets[i, action_t] = reward_t
                # Train our network using target and predicted Q values
                _, _loss = sess.run([train_op, loss], feed_dict={inputs:inputs_train, nextQ:targets})

            total_reward += reward
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2f}".format(episode, EPISODES, total_reward, EPSILON))
                EPSILON = 1.0/((episode/50) + 10) # Reduce chance of random action as we train the model.
                break

        rewards.append(total_reward)

        if episode > 0 and episode % 10 == 0:
            saver.save(sess, "save/frozen-dqn.ckpt")

print("Rewards: {:.2f}".format(sum(rewards)))
print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
