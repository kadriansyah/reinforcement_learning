import numpy as np
import random
import tensorflow as tf
import time

EPISODES = 100
GAMMA = 0.99
EPSILON = 0.1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
RENDER = True

bandits = [[-5, -1, 0, 1], [-1, -5, 1, 0], [0, 1, -1, -5]]
num_bandits = len(bandits)
num_actions = len(bandits[0])

tf.reset_default_graph()

# Placeholders
tf_state  = tf.placeholder(name='tf_state',  shape=[1], dtype=tf.int32)
tf_action = tf.placeholder(name='tf_action', shape=[1], dtype=tf.int32)
tf_reward = tf.placeholder(name='tf_reward', shape=[1], dtype=tf.float32)

# One hot encode the state
state_one_hot = tf.one_hot(indices=tf_state, depth=num_bandits)

# Feed forward net to choose the action
W = tf.Variable(tf.ones([num_bandits, num_actions]))
action_score = tf.nn.sigmoid(tf.matmul(state_one_hot, W))

chosen_weight = tf.slice(tf.reshape(action_score, [-1]), tf_action, [1])
loss = -(tf.log(chosen_weight) * tf_reward)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def pull_arm(bandit, action):
    """
    Pull the arm of a bandit and get a positive
    or negative result. (+/- 1)
    """

    # get random number from normal dist.
    answer = np.random.randn(1)

    # Get positive reward if bandit is higher than random result
    if bandit[action] > answer:
        return 1
    else:
        return -1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # # Restore model weights from previously saved model
    # saver.restore(sess, "save/m_armed_bandits/m_armed_bandits-dqn.ckpt")

    total_steps = 0
    rewards = np.zeros([num_bandits,num_actions])
    for episode in range(0, EPISODES):
        state = np.random.randint(0, num_bandits)
        # Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) <= EPSILON:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(sess.run(action_score, feed_dict={tf_state: [state]}))

        # Get new state and reward from environment
        reward = pull_arm(bandits[state], action)

        # train the model
        _ , weights = sess.run([train_op, W], feed_dict={tf_state:[state], tf_action:[action], tf_reward:[reward]})

        rewards[state][action] += reward
        if episode % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(rewards))

        EPSILON *= EPSILON_DECAY if EPSILON > EPSILON_MIN else 1

        for i in range(num_bandits):
            print("The agent thinks bandit-"+ str(i+1) +" arm: " + str(np.argmax(weights[i])+1) + " is the most promising....")
        print("")
        # if episode > 0 and (episode + 1) % 1000 == 0:
        #     saver.save(sess, "save/m_armed_bandits/m_armed_bandits-dqn.ckpt")
