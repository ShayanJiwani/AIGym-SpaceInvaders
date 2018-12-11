import gym
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from collections import deque
import sys
import random
env = gym.make('SpaceInvaders-v0')

#
# preprocess -> send to dense, fully connected, layer specifying shape

class InvaderNN():

    def __init__(self, env):
        self.env = env
        self.state = env.observation_space
        self.discount = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0
        self.epsilon_min = 0.01
        self.learning_rate = 0.01

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights( self.model.get_weights() )
        self.targets_data = []

        self.history_buffer = []
        self.history = []
        self.stack_size = 4



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
        input_buffer = []
        for instance in self.history_buffer:
            observation = cv2.resize(instance, dsize=(84, 110), interpolation=cv2.INTER_CUBIC)
            observation = observation[26::,]
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            input_buffer.append(observation)
        return np.array(input_buffer)
        # cv2.imshow('Window',observation)
        # cv2.waitKey(0)
        # print(observation)



invaderNN_model = InvaderNN(env)
scores = []
n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
outfile = open("log.csv","w")
outfile.write("episode,score,mean\n")
done = False

for i_episode in range(n_episodes):
    t = 0
    score = 0
    env.reset()
    done = False
    invaderNN_model.history_buffer = [env.step(5)[0] for _ in range(4)]
    curr_state = invaderNN_model.preprocess()

    while not done:

        initial_state = invaderNN_model.preprocess()
        invaderNN_model.history_buffer = []

        action = invaderNN_model.get_action(curr_state)


        if len(invaderNN_model.history_buffer) == 4:
            initial_state = invaderNN_model.preprocess()
            invaderNN_model.history_buffer = []

        action = invaderNN_model.get_action(curr_state)

        for i in range(4):
            #env.render()
            temp_observation, temp_score, temp_done, temp_info = env.step(action)
            score += temp_score
            invaderNN_model.history_buffer.append(temp_observation)
            done = done | temp_done
            t+=1

        new_state = invaderNN_model.preprocess()
        invaderNN_model.save_history((initial_state, action, score, done, new_state))
    # call fit here
    buf = []
    for i in range(len(invaderNN_model.history)):
        buf.append( (invaderNN_model.history[i][0],invaderNN_model.history[i][2])  )
        if len(buf) == 4:
            invaderNN_model.model.fit(np.array(buf[0]), np.array(invaderNN_model.targets_data), batch_size=1,epochs=10)
            buf = []

    scores.append(score)

    print("Episode {0} Score: {1} Mean: {2:0.2f} Epsilon: {3:0.2f} History: {4}".format(i_episode, score,np.mean(scores), invaderNN_model.epsilon, len(invaderNN_model.history)))
    outfile.write("{0},{1},{2}\n".format(i_episode, score, np.mean(scores)))
    invaderNN_model.save()

outfile.close()

xi = np.arange(n_episodes)
plt.figure()
plt.title("mean={}".format(np.mean(scores)))
plt.plot(xi,scores,'o', alpha=.3)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
plt.savefig("outfile.png")
#print([(hist['episode'], hist['action'],hist['reward']) for hist in invaderNN_model.history if hist['reward'] > 0])
