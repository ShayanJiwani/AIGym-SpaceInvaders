import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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
        self.e = 1
        self.lr = 0.01
        self.model_actual = self.create_model()
        self.model_target = self.create_model()
        self.history = []



        self.create_model()

    def set_state(self, observation):
        self.state = observation

    def save_history(self, history_item):
        self.history.append(history_item)

    def train(self):
        w = self.model.get_weights()
        target_w = self.target_model.get_weights()
        for i in range(len(target_w)):
            target_w[i] = w[i]
        self.model_target.set_weights(target_w)

    def create_model(self):
        model = Sequential()
        model.add( Dense(24, activation='relu', input_shape=(84,84) ) ) # (84,84,1)
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=SGD(lr=self.lr, clipnorm=1.))
        return model


    def fit_model(self):
        for instance in self.history:
            i_episode, action, state, reward, done, info = instance.values()
            target = self.model_target.predict(self.state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(state)[0])
                target[0][action] = reward + Q_future * self.discount
                self.model_actual.fit(state, target, epochs=1, verbose=0) # fit the actual model with data predicted by target model
        print(self.model_actual.get_weights())

    def get_action(self, state):
        # epsilon greedy
        if np.random.random() < self.e:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model_actual.predict(state)[0])


    # resize to (84,84)
    # grayscale
    # dimensions are (84,84,1)
    # remove top 26 rows as they only contain the score
    # concatenate the last four "images" of (84,84) to get 84x84*4 image as input
    def preprocess(self, observation):
        observation = cv2.resize(observation, dsize=(84, 110), interpolation=cv2.INTER_CUBIC)
        observation = observation[26::,]
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        self.state = observation
        # print(observation)
        # cv2.imshow('Window',observation)
        # cv2.waitKey(0)


    def get_action_list(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

invaderNN_model = InvaderNN(env)
scores = []
n_episodes = 20
for i_episode in range(n_episodes):

    observation = env.reset()

    #image_rescaled = rescale(observation, 1.0 / 4.0, anti_aliasing=False)
    #print(image_rescaled.shape)
    #break
    done = False
    prev_lives = 3
    t = 0
    score = 0
    while not done: # if not done after 1000 frames, game is stale
        t+=1
        env.render()
        action = invaderNN_model.get_action(observation) #env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        invaderNN_model.preprocess(new_observation)
        new_state = invaderNN_model.state

        if info['ale.lives'] < prev_lives:
            reward -= 50 * (3-info['ale.lives'])
            prev_lives -= 1

        score += reward*info['ale.lives']
        invaderNN_model.save_history( {'episode':i_episode, 'action': action, 'state': new_state, 'reward': reward, 'done':done, 'info':info} )
        invaderNN_model.fit_model()
        #invaderNN_model.train()
        if done:
            # score = 0
            # for h in invaderNN_model.history:
            #     score += h['reward']
            #print("Episode {} finished after {} timesteps with score {}".format(i_episode, t+1, score))
            break


    print("Episode {}: {}".format(i_episode, score))
    scores.append(score)

from scipy import stats

xi = np.arange(n_episodes)
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,scores)
line = slope*xi+intercept
plt.figure()
plt.plot(xi,scores,'o', xi, line)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
#print([(hist['episode'], hist['action'],hist['reward']) for hist in invaderNN_model.history if hist['reward'] > 0])
