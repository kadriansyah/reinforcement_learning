import gym
import numpy as np

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = 0.8
y = 0.95
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    # The Q-Table learning algorithm
    for time in range(500):
        # env.render()
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])

        rAll += r
        s = s1
        if d:
            print("episode: {}/{}, score: {}".format(i, num_episodes, rAll))
            break
    rList.append(rAll)
print("Rewards: {:.2f}".format(sum(rList)))
print("Percent of succesful episodes: {:.2f}%".format(sum(rList)/num_episodes))
