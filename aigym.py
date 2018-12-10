import gym
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from collections import deque
import sys
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
        # need to figure out what we are going to do with fit if we are only storing 4 frames at a time
        # we can't do batches if we are saving every 4 frames
        episodes, actions, states, new_states, rewards, dones, infos = np.random.choice(self.history_buffer, 4)
        print("episodes-", epsiodes, "...")
        print("action-", actions, "...")
        print("states-", states, "...")
        print("new_states-", new_states, "...")
        print("rewards-", rewards, "...")
        print("dones-", dones, "...")
        print("infos-", infos, "\n")
        targets = np.zeros((4, self.env.action_space.n))
        for i in range(32):
            data = s_batch[i].reshape(1,84,84,4)
            targets[i] = self.model.predict(states[i].reshape(1,84,84,4), batche_size=1)
            future_action = self.target_model.predict(new_states[i].reshape(1,84,84,4), batch_size=1)
            targets[i, actions[i]] = rewards[i]
            if not dones[i]:
                targets[i, actions[i]] += .99* np.max(future_action)

        loss = self.model.train_on_batch(states, targets)
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
    def preprocess(self, instance):
        observation = cv2.resize(instance, dsize=(84, 110), interpolation=cv2.INTER_CUBIC)
        observation = observation[26::,]
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        self.history_buffer.append(observation)
        if (len(self.history_buffer) == 4):
            return np.array(self.history_buffer)
        else:
            return None
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
outfile = open("log.csv","w")
outfile.write("episode,score,mean\n")

for i_episode in range(n_episodes):

    observation = invaderNN_model.env.reset()
    done = False
    prev_lives = 3
    t = 0
    score = 0
    while not done:

        t+=1
        #env.render()
        # state = invaderNN_model.preprocess() # the combined four state observation
        action = invaderNN_model.get_action(observation)
        new_observation, reward, done, info = env.step(action)
        score += reward
        # print("len of invader nn model history == ", len(invaderNN_model.history), "\n")
        # invaderNN_model.history_buffer.append(observation)
        # if state is not None:
        #     invaderNN_model.history_buffer.pop(0)
        # else:
        #     print("its none")
        invaderNN_model.preprocess(observation)
        #invaderNN_model.save_history( {'episode':i_episode, 'action': action, 'state':observation, 'new_state': new_observation, 'reward': reward, 'done':done, 'info':info} )
        # print("len is == ", len(invaderNN_model.history))
        if len(invaderNN_model.history_buffer) == 4:
            #print(invaderNN_model.history_buffer)
            invaderNN_model.fit()
            invaderNN_model.fit_target()
            invaderNN_model.history_buffer = []

        observation = new_observation

        if done:break
    #invaderNN_model.fit_model() # fit data from the episode
    print("Episode {0} Score: {1} Mean: {2:0.2f}".format(i_episode, score,np.mean(scores)))
    outfile.write("{0},{1},{2}\n".format(i_episode, score, np.mean(scores)))
    scores.append(score)
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
