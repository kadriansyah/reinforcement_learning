import gym
import numpy as np
import pickle
from pathlib import Path

env = gym.make('FrozenLake8x8-v0')

# Initialize table with all zeros
if Path('save/frozen/frozen8x8-q-table.ckpt').is_file():
    q_table = open('save/frozen/frozen8x8-q-table.ckpt','rb')
    Q = pickle.load(q_table)
else:
    Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
alpha = 0.8
gamma = 0.99
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rewards = []
for episode in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    total_reward = 0
    # The Q-Table learning algorithm
    for time in range(500):
        env.render()
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))

        # Get new state and reward from environment
        next_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge
        Q[state,action] = Q[state,action] + alpha*(reward + gamma*np.max(Q[next_state,:]) - Q[state,action])

        total_reward += reward
        state = next_state
        if done:
            print("episode: {}/{}, score: {}".format(episode + 1, num_episodes, total_reward))
            break
    rewards.append(total_reward)
    if (episode + 1) % 500 == 0:
        # save Q-Table
        print('[episode{:d}] saving Q-Table...'.format(episode + 1))
        q_table = open('save/frozen/frozen8x8-q-table.ckpt','wb')
        pickle.dump(Q, q_table)
        q_table.close()
print("Rewards: {:.2f}".format(sum(rewards)))
print("Percent of succesful episodes: {:.2f}%".format((sum(rewards)/num_episodes) * 100))
