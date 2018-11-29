import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import cv2

env = gym.make('SpaceInvaders-v0')

#
# preprocess -> send to dense, fully connected, layer specifying shape

class InvaderNN():

    def __init__(self, env):
        self.env = env
        self.state = None
        self.model = None
        self.history = []
        self.create_model()

    def save_history(self, history_item):
        self.history.append(history_item)

    def train(self):
        pass

    def create_model(self):
        model = Sequential()
        model.add( Dense(64, activation='relu', input_shape=(84,84,1) ) )
        model.add(Dense(64, activation='relu'))
        model.add(Dense(env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=SGD(lr=0.01, clipnorm=1.))
        self.model = model

    def get_action(self):
        # epsilon greedy
        self.model.get_weights()


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
        return observation
        # cv2.imshow('Window',observation)
        # cv2.waitKey(0)


    def get_action_list(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

invaderNN  = InvaderNN(env)
for i_episode in range(2):

    observation = env.reset()
    #image_rescaled = rescale(observation, 1.0 / 4.0, anti_aliasing=False)
    #print(image_rescaled.shape)
    #break
    done = False
    prev_lives = 3
    while not done:

        env.render()
        action = 1#env.action_space.sample() # invaderNN.get_action()
        new_observation, reward, done, info = env.step(action)
        new_state = invaderNN.preprocess(new_observation)
        invaderNN.save_history( {'action': action, 'state': new_state, 'reward': reward, 'done':done, 'info':info} )

        if info['ale.lives'] < prev_lives:
            reward -= 50 * (3-info['ale.lives'])
            prev_lives -= 1

        if done:
        #     print("Episode finished after {} timesteps".format(t+1))
            break
print([(hist['action'],hist['reward']) for hist in invaderNN.history if hist['reward'] > 0])
