import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt

env = gym.make('SpaceInvaders-v0')

#
# preprocess -> send to dense, fully connected, layer specifying shape

class InvaderNN():

    def __init__(self, env):
        self.env = env
        self.state = env.observation_space
        self.discount = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.model = self.build_model()
        self.history = []


    def set_state(self, observation):
        self.state = observation

    def save_history(self, history_item):
        self.history.append(history_item)

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_dim=84*84 )) # (84,84,1)
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model


    def fit_model(self):

        if len(self.history) < 32:
            return
        batch = np.random.choice(self.history, 32)

        for sample in batch:
            target = sample['reward']
            if sample['done']:
                q_value = self.model.predict(sample['new_state'])[0]
                target += self.discount * np.argmax(q_value )
            target_f = self.model.predict(sample['state'])
            target_f[0][sample['action']] = target
            self.model.fit(sample['state'], target_f, epochs=1, verbose=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_action(self, state):
        # epsilon greedy

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    # resize to (84,84)
    # grayscale
    # dimensions are (84,84,1)
    # remove top 26 rows as they only contain the score
    # concatenate the last four "images" of (84,84) to get 84x84*4 image as input
    def preprocess(self, observation):
        observation = cv2.resize(observation, dsize=(84, 110), interpolation=cv2.INTER_CUBIC)
        observation = observation[26::,]
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return observation
        # print(observation)
        # cv2.imshow('Window',observation)
        # cv2.waitKey(0)


invaderNN_model = InvaderNN(env)
scores = []
n_episodes = 100

for i_episode in range(n_episodes):

    observation = invaderNN_model.preprocess( env.reset() ).reshape((1, 84*84))
    done = False
    prev_lives = 3
    t = 0
    score = 0
    while not done:
        t+=1
        env.render()
        action = invaderNN_model.get_action(observation) #env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        new_observation = invaderNN_model.preprocess(new_observation).reshape((1, 84*84))

        # curr_lives = info['ale.lives']
        # if curr_lives < prev_lives:
        #     reward -= 50 * (3-curr_lives)
        #     prev_lives -= 1

        score += reward
        invaderNN_model.save_history( {'episode':i_episode, 'action': action, 'state':observation, 'new_state': new_observation, 'reward': reward, 'done':done, 'info':info} )

        observation = new_observation

        if done:break
    invaderNN_model.fit_model() # fit data from the episode
    print("Episode {}: {}".format(i_episode, score))
    scores.append(score)

from scipy import stats

xi = np.arange(n_episodes)
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,scores)
line = slope*xi+intercept
plt.figure()
plt.title("y={0:0.2f}*x+{1:0.2f}".format(slope, intercept))
plt.plot(xi,scores,'o', xi, line)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
#print([(hist['episode'], hist['action'],hist['reward']) for hist in invaderNN_model.history if hist['reward'] > 0])
