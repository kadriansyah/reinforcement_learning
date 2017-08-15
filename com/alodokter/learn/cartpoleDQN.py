import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

REPLAY_MEMORY = 500 # number of previous transitions to remember
EPISODES = 1000
OBSERVATION = 100 # timesteps to observe before training

class CartPoleDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate)) # Configures the model for training
        return model

    # experience replay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # act_values format is ndarray([[score-1, score-2]])
        return np.argmax(act_values[0]) # returns action index (index of maximum action's score)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Trains the model for a fixed number of epochs (iterations on a dataset)
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # state_size = 4
    # position (x), velocity (x_dot), angle (theta), and angular velocity (theta_dot)
    state_size = env.observation_space.shape[0]

    # action_size = 2 (move to the left or move to the right)
    action_size = env.action_space.n

    agent = CartPoleDQN(state_size, action_size)
    agent.load("save/cartpole-dqn.ckpt")

    rewards = []
    batch_size = 32
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            # env.render()
            action = agent.choose_action(state) # find the best action from this state
            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > REPLAY_MEMORY:
                agent.memory.popleft()

            # # experience replay
            # if len(agent.memory) > batch_size:
            #     agent.replay(batch_size)

            total_reward += reward
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2f}".format(episode, EPISODES, time, agent.epsilon))
                break

        rewards.append(total_reward)

        # experience replay (* should be inside time loop)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if episode > 0 and episode % 10 == 0:
            agent.save("save/cartpole-dqn.ckpt")

# print("Rewards: {:.2f}".format(sum(rewards)))
# print("Percent of succesful episodes: {:.2f}%".format(sum(rewards)/EPISODES))
