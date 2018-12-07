import gym
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from collections import deque
env = gym.make('SpaceInvaders-v0')
import sys

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
        self.target_model = self.build_model()
        self.target_model.set_weights( self.model.get_weights() )

        self.history_buffer = []
        self.history = deque(maxlen=200)
        self.stack_size = 4
        self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
        #self.load()


    def set_state(self, observation):
        self.state = observation

    def save_history(self, history_item):
        self.history.append(history_item)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, 4)))
        model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def fit(self):
        batch = np.random.sample(self.history, 32)
        targets = np.zeros((4, self.env.action_space.n))
        for i in range(4):
            data = s_batch[i].reshape(1,84,84,4)
            targets[i] = self.model.predict(data)
            future_action = self.target_model.predict(new_state_batch[i].reshape(1,84,84,4))
        loss = self.model.train_on_batch(state_batch, targets)
        print("loss =", loss)

    def fit_target(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = .1 * model_weights[i] + (.9) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    def get_action(self, state):
        # epsilon greedy

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            data = state.reshape(1,84,84,4)
            q_actions = self.model.predict(data, batch_size=1)
            return np.argmax(q_actions) # optimal policy

    # resize to (84,84)
    # grayscale
    # dimensions are (84,84,1)
    # remove top 26 rows as they only contain the score
    # concatenate the last four "images" of (84,84) to get 84x84*4 image as input
    def preprocess(self):
        if len(self.history_buffer) < 4:
            return None
        input_buffer = []
        for instance in self.history_buffer:
            observation = cv2.resize(instance, dsize=(84, 110), interpolation=cv2.INTER_CUBIC)
            observation = observation[26::,]
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            input_buffer.append(observation)
        return np.array(input_buffer)
        # print(observation)
        # cv2.imshow('Window',observation)
        # cv2.waitKey(0)


    def save(self):
        model_json = self.model.to_json()
        with open("model_num.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model_num.h5")

    def load(self):
        json_file = open('model_num.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_num.h5")


invaderNN_model = InvaderNN(env)
scores = []
n_episodes = int(sys.argv[1]) or 100

for i_episode in range(n_episodes):

    observation = invaderNN_model.env.reset()
    done = False
    prev_lives = 3
    t = 0
    score = 0
    while not done:
        t+=1
        #env.render()
        state = invaderNN_model.preprocess() # the combined four state observation

        action = invaderNN_model.get_action(observation)
        new_observation, reward, done, info = env.step(action)
        score += reward

        invaderNN_model.history_buffer.append(observation)
        if state is not None:
            invaderNN_model.history_buffer.pop(0)

        invaderNN_model.save_history( {'episode':i_episode, 'action': action, 'state':observation, 'new_state': new_observation, 'reward': reward, 'done':done, 'info':info} )

        if len(invaderNN_model.history_buffer) > 32:
            invaderNN_model.fit()
            invaderNN_model.fit_target()

        observation = new_observation

        if done:break
    #invaderNN_model.fit_model() # fit data from the episode
    print("Episode {}: {}".format(i_episode, score))
    scores.append(score)
    invaderNN_model.save()


from scipy import stats
from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x +b


xi = np.arange(n_episodes)
popt, pconv = curve_fit(f, xi, score)
line = f(xi, *popt)

plt.figure()
plt.title("y={0:0.2f}*x + {1:0.2f}".format(popt[0], popt[1]))
plt.plot(xi,scores,'o', xi, line)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
plt.savefig("outfile.png")
#print([(hist['episode'], hist['action'],hist['reward']) for hist in invaderNN_model.history if hist['reward'] > 0])
